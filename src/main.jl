#ENV["JULIA_CUDA_USE_COMPAT"] = false #uncomment if there is problems with initialising CUDA related packages
#using MKL
using Base: Float64
T = Float32;
using PyCall
using PyPlot
plt.rc("font", family="sans")  # Use generic fallback
using DelimitedFiles
using WriteVTK
using TimerOutputs
using JLD2
using Glob
using CUDA

#identify least used GPU and switch to that
devs = CUDA.devices()
mem_free = Float64[]

for (i,dev) in enumerate(devs)
    CUDA.device!(dev)

    # This forces context initialization
    CUDA.zeros(1)

    # Get free and total memory in bytes
    free, total = CUDA.memory_info()
    push!(mem_free, free)

    println("Device $(i -1): ", 
            " - Free memory: ", round(free / 1_048_576, digits=2), " MB / ",
            round(total / 1_048_576, digits=2), " MB")
end

# Select device with most free memory
best_idx = argmax(mem_free)
best_dev = collect(devs)[best_idx]

CUDA.device!(best_dev)
println("Selected GPU: $(best_idx - 1)")

include("settings.jl")
include("SolverGPU.jl")
include("SolverCPU.jl")

#specify toml file location
file_path1 = "configFiles/config_SingleBeam_Homogeneous_Boltzmann.toml" 
file_path2 = "configFiles/config_SingleBeam_Homogeneous_FP.toml"
file_path3 = "configFiles/config_SingleBeam_Heterogeneous_Boltzmann.toml"
file_path4 = "configFiles/config_SingleBeam_Heterogeneous_FP.toml"
file_path5 = "configFiles/config_SeveralBeams_Homogeneous_Boltzmann.toml"
file_path6 = "configFiles/config_SeveralBeams_Homogeneous_FP.toml"

# Global state for function parameters
const dqage_state = Dict{Symbol, Any}()
const to = TimerOutput()

function runAndPlot(file_path)
    config = TOML.parsefile(file_path)
    trace = get(config["computation"], "trace", "false")
    disableGPU = get(config["computation"], "disableGPU", "false")
    mu_e = get(config["physics"], "eKin", 90)
    file_name = split(file_path, ".")[1]
    order = get(config["numerics"], "order", 2)

    close("all")
    
    info = "CUDA"
    @timeit to "Geometry and physics set-up" begin
        s = Settings(file_path);
        if CUDA.functional() && ~disableGPU
            solver1 = SolverGPU(s,order);
        else
            solver1 = SolverCPU(s,order);
        end
    end
    println("rMax = $(s.rMax)")
    println("r = $(s.r)")
    @timeit to "Solver" begin
        X_dlr,S_dlr,W_dlr,_, dose_DLR, dose_coll, rankInTime₂, ψ₂ = getfield(Main,Symbol("Solve$(s.solverName)"))(solver1,s.model,trace);
    end
    println("Average rank = $(mean(rankInTime₂[2,2:end]))")
    u = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,X_dlr*Diagonal(S_dlr)*W_dlr[1,:]);
    dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);
    dose_coll = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_coll);

    println(to)
    idxX = Int(ceil(s.NCellsX/2))
    idxY = Int(ceil(s.NCellsY/2))   
    idxZ = Int(ceil(s.NCellsZ/2))

    X = (s.xMid'.*ones(size(s.yMid)))
    Y = (s.yMid'.*ones(size(s.xMid)))'
    Z = (s.zMid'.*ones(size(s.yMid)))
    XZ = (s.xMid'.*ones(size(s.zMid)))
    ZX = (s.zMid'.*ones(size(s.xMid)))
    YZ = (s.yMid'.*ones(size(s.zMid)))'

    save("output/rankInEnergy_nPN$(s.nPN)_tol$(s.epsAdapt)_$(s.tracerFileName).jld2", "energy", solver1.csd.eGrid[2:end], "rank", rankInTime₂[2,:])
    save("output/dose_nPN$(s.nPN)_tol$(s.epsAdapt)_$(s.tracerFileName)_$(file_name).jld2", "dose", dose_DLR./sum(dose_DLR[dose_DLR.>0]) * sum(mu_e),"x", s.xMid,"y",s.yMid,"z",s.zMid)

    epsAdapt=s.epsAdapt
    #epsAdapt1=s1.epsAdapt
    fig = figure()
    ax = gca()
    ltype = ["b-","r--","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--","b-","r-","m-","g-","y-","k-","b--","r--","m--","g--","y--","k--"]
    labelvec = ["1st order","2nd order",L"\vartheta = $s.epsAdapt"]
    ax.plot(rankInTime₂[1,2:end].-s.eRest,rankInTime₂[2,2:end], "-g", label="ϑ=$epsAdapt",linewidth=2, alpha=1.0)
    ax.set_xlabel("pseudo time", fontsize=20);
    ax.set_ylabel("rank", fontsize=20);
    ax.tick_params("both",labelsize=20) 
    ax.legend(loc="upper left", fontsize=20)
    fig.canvas.draw() # Update the figure
    savefig("output/ranks_$(s.tracerFileName)_tol$(s.epsAdapt).png")

    # line plot dose
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, dpi=200);
    #nyRef = length(yRef)
    im1 = ax1.pcolormesh(YZ',Z',dose_DLR[idxX,:,:]',vmin=0,vmax=maximum(dose_DLR[idxX,:,:]),cmap="jet")
    plt.colorbar(im1,ax=ax1)
    im2 = ax2.pcolormesh(YZ',Z',dose_coll[idxX,:,:]',vmax=maximum(dose_coll[idxX,:,:]),cmap="jet")
    plt.colorbar(im2,ax=ax2)
    im3 = ax3.pcolormesh(YZ',Z',dose_DLR[idxX,:,:]'.-dose_coll[idxX,:,:]',vmin=0,vmax=maximum(dose_DLR[idxX,:,:].-dose_coll[idxX,:,:]),cmap="jet")
    plt.colorbar(im3,ax=ax3)
    ax4.plot(s.zMid,dose_DLR[idxX,idxY,:],label="total dose")
    ax4.plot(s.zMid,dose_coll[idxX,idxY,:],label="collided")
    ax4.plot(s.zMid,dose_DLR[idxX,idxY,:] .- dose_coll[idxX,idxY,:],label="uncollided")
    #load and plot reference results

    ax1.title.set_text("Total dose")
    ax2.title.set_text("Collided dose")
    ax3.title.set_text("Uncollided dose")
    ax4.title.set_text("Dose cut along z-axis")
    ax4.legend()
    ax1.axis("equal")
    ax2.axis("equal")
    ax3.axis("equal")
    #ax4.axis("equal")
    show()
    tight_layout()
    savefig("output/Results_$(s.nPN)_$(s.tracerFileName).png")
    
    # # # #plot angular basis
    # file_names = glob("Wdlr_$(s.tracerFileName)*.txt")
    # for i=1:length(file_names)
    #     file_name_Wdlr = split.(file_names[i], ".")[1]
    #     run(`python plotW_new.py "$file_name_Wdlr"`)
    # end

    println("main for config $(file_path) finished")
end

runAndPlot(file_path1)
# runAndPlot(file_path2)
# runAndPlot(file_path3)
# runAndPlot(file_path4)
# runAndPlot(file_path5)
# runAndPlot(file_path6)