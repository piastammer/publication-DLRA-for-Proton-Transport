__precompile__
using Interpolations
include("MaterialParametersProtons.jl")

struct CSD{T<:AbstractFloat}
    # energy grid
    eGrid::Array{T,1};
    # transformed energy grid
    eTrafo::Array{T,1};
    # stopping power for computational energy grid
    S::Array{T,2};
    SMid::Array{T,2};
    # tabulated energy for sigma/stopping power
    E_Tab::Array{T,1};
    # tabulated sigma
    sigma_ce::Array{T,3};
    sigma_xi::Array{T,3};
    # settings
    settings::Settings

    # constructor
    function CSD(settings::Settings,T::DataType=Float64)
        # read tabulated material parameters
        if settings.particle =="Protons"
            param = MaterialParametersProtons(settings::Settings,settings.OmegaMin);
            S_tab = param.S_tab;
            E_tab = param.E_tab;
            sigma_ce = param.sigma_ce;
            sigma_xi = param.sigma_xi;
        else #Electrons (not supported at the moment!!)
            println("Only protons supported in this TITUS version!")
        end
        nTab = length(E_tab)
        E_transformed = zeros(nTab)
        for i = 2:nTab
            E_transformed[i] = E_transformed[i - 1] + ( E_tab[i] - E_tab[i - 1] ) / 2 * ( 1.0 / S_tab[i] + 1.0 / S_tab[i - 1] );
        end

        # define minimal and maximal energy for computation
        minE = settings.eMin .+ settings.eRest;
        maxE = settings.eMax;

        eTrafoMax = integrate(E_tab, 1 ./S_tab[:,1])
        eTrafo1 = zeros(nTab)
        for i = 1:length(E_tab)
            eTrafo1[i] = eTrafoMax  - integrate(E_tab[1:i], 1 ./S_tab[1:i,1])
        end
        
        ETab2ETrafo = LinearInterpolation(E_tab, eTrafo1; extrapolation_bc=Flat())
        eMaxTrafo = ETab2ETrafo( maxE );
        eMinTrafo = ETab2ETrafo( minE );
        nEnergies = Integer(ceil(maxE/settings.dE));
        if ~iseven(nEnergies)
            nEnergies = nEnergies + 1;
        end
        eGrid = collect(range(minE,maxE,length=nEnergies))[end:-1:1]
        dEGrid=zeros(length(eGrid)-1)
        for i=2:length(eGrid)
            dEGrid[i-1] = eGrid[i-1] - eGrid[i]
        end
        ETrafo2ETab = LinearInterpolation(eTrafo1[end:-1:1], E_tab[end:-1:1].-settings.eRest; extrapolation_bc=Flat())
        eTrafo = ETab2ETrafo(eGrid)

        # compute stopping power for computation
        if size(S_tab,2) == 1 #only one material/waterequivalent
            S = zeros(size(eGrid,1),1)
            SMid = zeros(size(eGrid,1),1)

            E2S = LinearInterpolation(E_tab, S_tab[:,1]; extrapolation_bc=Flat())
            S[:,1] = E2S(eGrid)
            # compute stopping power at intermediate time points
            dE = zeros(length(eTrafo)-1)
            for i=1:length(eTrafo)-1
                dE[i] = eTrafo[i+1]-eTrafo[i];
            end
            SMid[:,1] = E2S(eGrid[1:end].-0.5*dEGrid[1])

            eGridMid = ETrafo2ETab(eMaxTrafo .- (eTrafo[1:(end-1)].+0.5.*dE))
        else
            nPsi = size(S_tab,2)
            E2S = interpolate((E_tab,1:nPsi), S_tab,(Gridded(Linear()),NoInterp()))
            S=zeros(nEnergies,nPsi)
            for i = 1:nEnergies
                S[i,:] = E2S.(eGrid[i],1:nPsi)
            end
            # compute stopping power at intermediate time points
            dE = zeros(length(eTrafo)-1)
            for i=1:length(eTrafo)-1
                dE[i] = eTrafo[i+1]-eTrafo[i];
            end
            SMid=zeros(nEnergies-1,nPsi)
            for i = 1:nEnergies-1
                SMid[i,:] = E2S.(eGrid[i].-0.5*dEGrid[i],1:nPsi)
            end
        end
        new{T}(eGrid,eTrafo,S,SMid,E_tab,sigma_ce,sigma_xi,settings);
    end
end

function XiAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_xi,2)
    y = zeros(2,nPsi)
    E2Sigma_xi1 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,1],(Gridded(Linear()),NoInterp()))
    E2Sigma_xi2 = interpolate((obj.E_Tab,1:nPsi), obj.sigma_xi[:,:,2],(Gridded(Linear()),NoInterp()))
    y[1,:] = E2Sigma_xi1.(energy,1:nPsi)
    y[2,:] = E2Sigma_xi2.(energy,1:nPsi)
    return T.(y);
end

function SigmaAtEnergyandX(obj::CSD{T}, energy::T) where {T<:AbstractFloat}
    nPsi = size(obj.sigma_ce,3)
    y = zeros(obj.settings.nPN+1,nPsi)
    for i = 1:(obj.settings.nPN+1)
        # define Sigma mapping for interpolation at moment i
        E2Sigma_ce = interpolate((obj.E_Tab,1:nPsi), obj.sigma_ce[:,i,:],(Gridded(Linear()),NoInterp()))
        y[i,:] = E2Sigma_ce.(energy,1:nPsi);
    end
    return T.(y);
end

function setup_RTEnergyGrps(no_grps, E_min, E_max)
    E_grps = zeros(no_grps+1)
    dE = (E_max - E_min) / no_grps
    for gr=1:(no_grps+1)
    E_grps[no_grps-gr+2] = E_min + (gr-1) * dE
    end
    return E_grps
end

function computeOutscattering(obj::CSD{T}, energy::Array{T,1},minOmega,type::String)
    param = MaterialParametersProtons(obj.settings,minOmega);
    E_tab = param.E_tab;
    sigma_ce = param.sigma_ce;
    xi=zeros(length(energy),12)

    if type == "gaussIntTracer"
        nE = size(energy,1)        
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        if minOmega > 0
            w[mu.>cosd(minOmega)] .= 0;
        end
        nMat = 12
        N = size(obj.sigma_ce,2)
        open("tracer/proton_totalXS_data", "w") do file
            println(file, nE)
            for i=1:nE
                println(file, energy[i]-938.26)
            end
            for k=1:nMat
                println(file,matNames[k])
                for n = 1:nE
                    xi_e = 2*pi*dot(w,sigma_ce[n,:,k])
                    println(file,xi_e) 
                end
            end
        end
    elseif type == "gaussInt" #doesnt write files for tracer
        nE = size(energy,1)
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        nMat = 12
        for k=1:nMat
            for n = 1:nE
                xi[n,k] = 2*pi*dot(w,sigma_ce[n,:,k])
            end
        end
    elseif type == "FP_correction"
        N=19
        nE = size(energy,1)
        mu, w = gausslegendre(50);
        sigma_ce, matNames = Sigma_eModels_perMat(energy.-938.26,mu,1)
        nMat = 12
        N = size(obj.sigma_ce,2)
        open("tracer/proton_totalXS_data", "w") do file
            println(file, nE)
            for i=1:nE
                println(file, energy[i]-938.26)
            end
            for k=1:nMat
                println(file,matNames[k])
                for n = 1:nE
                    xi_e = 2*pi*dot(w,sigma_ce[n,:,k].*(1 .- mu)) 
                    println(file,0.5*xi_e*(N*(N+1))) 
                end
            end
        end
    end
    return xi*param.comp_vector
end
