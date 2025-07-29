__precompile__

using Images, FileIO, TOML, Meshes
include("utils.jl")
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    Nz::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    NCellsZ::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    e::Float64;
    f::Float64;
    # grid cell width
    dx::Float64
    dy::Float64
    dz::Float64

    # time settings
    # end time
    eMax
    eMin
    eRest
    # time increment
    dE::Float64;
    # CFL number 
    cfl::Float64;
    #number beam energies
    N_E
    
    # degree PN
    nPN::Int64;

    # spatial grid
    x
    xMid
    y
    yMid
    z
    zMid

    # problem definitions
    problem::String;

    #particle type
    particle::String;
    # beam properties
    x0 #::Array{Float64,1};
    y0 #::Array{Float64,1};
    z0 #::Array{Float64,1};
    Omega1 #::Array{Float64,1};
    Omega2 #::Array{Float64,1};
    Omega3 #::Array{Float64,1};
    OmegaMin::Float64;
    densityMin::Float64;
    sigmaX::Float64; # spatial std of initial beam
    sigmaY::Float64; # spatial std of initial beam
    sigmaZ::Float64; # spatial std of initial beam
    sigmaE::Float64; # energy std of boundary beam

    # physical parameters
    sigmaT::Float64;
    sigmaS::Float64;    

    # patient density
    density::Array{Float64,3};
    densityHU::Array{Float64,3};

    # rank
    r::Int; #for adaptive this is initial rank
    rMax::Int; #and this is the max allowed rank

    sizeOfTracerCT::Array{Int,1};

    # tolerance for rank adaptivity
    epsAdapt::Float64;  
    adaptIndex::Float64;

    #file names
    solverName::String;
    tracerFileName::String;
    model::String;

    function Settings(filePath::String)
        # load config
        config = TOML.parsefile(filePath)
        particle = get(config["physics"], "particle", "Protons")
        problem = get(config["physics"], "problem", "BoxInsert")
        model = get(config["physics"], "model", "Boltzmann")
        OmegaMin = get(config["physics"], "OmegaMin", 0)
        mu_e = get(config["physics"], "eKin", 90)
        Nx = get(config["numerics"], "nx", 43)
        Ny = get(config["numerics"], "ny", 43)
        Nz = get(config["numerics"], "nz", 163)
        r = get(config["numerics"], "rank", 20)
        rMax = get(config["numerics"], "maxRank", 100)
        order = get(config["numerics"], "order", 2)
        nPN = get(config["numerics"], "nMoments", 95)
        solverName = get(config["numerics"], "solverName", "Tracer_rankAdaptiveInEnergy")
        tracerFileName = get(config["numerics"], "tracerFileName", "eDep_$(problem)_$(model).bin")
        cfl = get(config["numerics"], "cfl", 10)
        epsAdapt = get(config["numerics"], "tolerance", 0.01)
        non_uniform = get(config["numerics"], "gridAdapt", false)
        
        #Proton rest energy
        if particle == "Protons"
            eRest = 938.26 #MeV
        elseif particle == "Electrons"
            eRest = 0.5 #MeV -> not used here
            println("Only protons supported in this version of TITUS!")
        end
        # spatial grid setting
        if order ==1
            NCellsX = Nx - 1;
            NCellsY = Ny - 1;
            NCellsZ = Nz - 1;
        elseif order == 2
            NCellsX = Nx - 3;
            NCellsY = Ny - 3;
            NCellsZ = Nz - 3;
        end

        a = 0.0; # left boundary
        b = 2.0; # right boundary

        c = 0.0; # lower boundary
        d = 2.0; # upper boundary

        e = 0.0; # left z boundary
        f = 7.0; # right z boundary

        density = ones(NCellsX,NCellsY,NCellsZ); 
        densityHU = zeros(NCellsX,NCellsY,NCellsZ); #HU

        # physical parameters
        sigmaS = 0.0;
        sigmaA = 0.0;
        eMax = 90.0;
        x0 = 0.5*b;
        y0 = 0.5*d;
        z0 = 0.0*f;
        Omega1 = 0.0;
        Omega2 = 0.0;
        Omega3 = 1.0;
        densityMin = 0.2;
        adaptIndex = 1;
        sigmaX = 0.1;
        sigmaY = 0.1;
        sigmaZ = 0.01;
        eMin = 0.001;
        sizeOfTracerCT = [NCellsX,NCellsY,NCellsZ]; #default is same as DLRA grid
        nE_perBeam = [0; 1]
        
        if problem == "BoxInsert"
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 2.0; # upper boundary
            e = 0.0;
            f = 7.0;
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = 0.0;
            Omega2 = 0.0;
            Omega3 = 1.0;
            x0 = 0.5 * b;
            y0 = 0.5 * d;
            z0 = 0.0 * f;
            density[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= 0.6190303991130821; #inserted box of lower density 
            densityHU[:,1:Int(ceil(NCellsY*0.5)),Int(floor(NCellsZ*0.3))+1:Int(floor(NCellsZ*0.6))] .= -400; #inserted box of lower density defined in HU
        elseif problem == "TwoBeams"
            nB=2;
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 4.0; # upper boundary
            e = 0.0;
            f = 4.0;
            w_e = 1
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = maximum(mu_e) + 5*maximum(sigmaE); 
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = zeros(nB)
            Omega2 = zeros(nB)
            Omega3 = zeros(nB)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)
            Omega1[1] = 0.0;
            Omega2[1] = 0.0;
            Omega3[1] = 1.0;

            # #90°
            Omega1[2] = 0.0;
            Omega2[2] = -1.0;
            Omega3[2] = 0.0;

            x0[1] = 0.5 * b;
            y0[1] = 0.5 * d;
            z0[1] = 0.0 * f;

            # #for 90°/60°
            x0[2] = 0.5 * b;
            y0[2] = 0.0 * d;
            z0[2] = 0.5 * f;
        elseif problem == "SingleBeam"
            nB=1;
            a = 0.0; # left boundary
            b = 2.0; # right boundary
            c = 0.0; # lower boundary
            d = 2.0; # upper boundary
            e = 0.0;
            f = 7.0;
            #mu_e = 50;
            w_e = 1;
            N_E = length(mu_e)
            sigmaE = mu_e * 1/100; #set to 1% of the beam energy
            eKin = mu_e + 5*sigmaE; #maximum energy mean plus five standard devs
            eMax = eKin + eRest 
            eMin = 0.011;
            sigmaX = 0.3;
            sigmaY = 0.3;
            sigmaZ = 0.01; 
            sigmaS = 1;
            sigmaA = 0.0;  
            adaptIndex = 1;
            Omega1 = zeros(nB)
            Omega2 = zeros(nB)
            Omega3 = zeros(nB)
            x0 = zeros(nB)
            y0 = zeros(nB)
            z0 = zeros(nB)

            Omega1[1] = 0.0;
            Omega2[1] = 0.0;
            Omega3[1] = 1.0;

            x0[1] = 0.5 * b;
            y0[1] = 0.5 * d;
            z0[1] = 0.0 * f;
        end
        sigmaT = sigmaA + sigmaS;

        if order == 2
            # Initialize the grid with ghost cells for second-order accuracy
            x = collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];

            # Add two ghost cells on each boundary
            x = [x[1] - 2*dx; x[1] - dx; x; x[end] + dx]
            y = [y[1] - 2*dy; y[1] - dy; y; y[end] + dy]
            z = [z[1] - 2*dz; z[1] - dz; z; z[end] + dz]

            # Calculate the cell boundaries by shifting by half a grid spacing
            x = x .+ dx/2
            y = y .+ dy/2
            z = z .+ dz/2

            # Calculate the midpoints of the cells
            xMid = x[2:(end-2)] .+ 0.5 * dx
            yMid = y[2:(end-2)] .+ 0.5 * dy
            zMid = z[2:(end-2)] .+ 0.5 * dz
        else
            x = collect(range(a, stop=b, length=NCellsX))
            dx = x[2] - x[1]
            y = collect(range(c, stop=d, length=NCellsY))
            dy = y[2] - y[1]
            z = collect(range(e, stop=f, length=NCellsZ))
            dz = z[2]-z[1];
            x = [x[1]-dx;x]; # add ghost cells so that boundary cell centers lie on a and b
            x = x.+dx/2;
            xMid = x[1:(end-1)].+0.5*dx
            y = collect(range(c,stop = d,length = NCellsY));
            y = [y[1]-dy;y]; # add ghost cells so that boundary cell centers lie on a and b
            y = y.+dy/2;
            yMid = y[1:(end-1)].+0.5*dy
            z = collect(range(e,stop = f,length = NCellsZ));
            z = [z[1]-dz;z]; # add ghost cells so that boundary cell centers lie on a and b
            z = z.+dz/2;
            zMid = z[1:(end-1)].+0.5*dz
        end

        # time settings
        dE = cfl*min(dx,dy,dz)

        sigmaE = maximum(sigmaE)
        # build class
        new(Nx,Ny,Nz,NCellsX,NCellsY,NCellsZ,a,b,c,d,e,f,dx,dy,dz,eMax,eMin,eRest,dE,cfl,N_E,nPN,x,xMid,y,yMid,z,zMid,problem,particle,x0,y0,z0,Omega1,Omega2,Omega3,OmegaMin,densityMin,sigmaX,sigmaY,sigmaZ,sigmaE,sigmaT,sigmaS,density,densityHU,r,rMax,sizeOfTracerCT,epsAdapt,adaptIndex,solverName,tracerFileName,model);
    end
end
