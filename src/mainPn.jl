#ENV["JULIA_CUDA_USE_COMPAT"] = false #uncomment if there is problems with initialising CUDA related packages
#using MKL
using Base: Float64
T = Float32;
using PyCall
using PyPlot
using DelimitedFiles
using WriteVTK
using TimerOutputs
using JLD2
using CUDA

include("settings.jl")
#include("SolverGPU.jl")
include("SolverCPU.jl")

# # specify toml file location
# file_path = "configFiles/config_FullPn_SingleBeam_Homogeneous_Boltzmann.toml" 
# file_path = "configFiles/config_FullPn_SingleBeam_Heterogeneous_Boltzmann.toml" 
file_path = "configFiles/config_FullPn_SingleBeam_Homogeneous_FP.toml" 
# file_path = "configFiles/config_FullPn_SingleBeam_Heterogeneous_FP.toml" 

file_name = split(file_path, ".")[1]
# Global state for function parameters
const dqage_state = Dict{Symbol, Any}()
const to = TimerOutput()

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
    rhoMin = minimum(s.density);
    if CUDA.functional() && ~disableGPU
        solver1 = SolverGPU(s,order);
    else
        solver1 = SolverCPU(s,order);
    end
end
println("rMax = $(s.rMax)")
println("r = $(s.r)")
@timeit to "Solver" begin
    dose_DLR, dose_coll = getfield(Main,Symbol("Solve$(s.solverName)"))(solver1,s.model,trace);
end

dose_DLR = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_DLR);
dose_coll = Vec2Ten(s.NCellsX,s.NCellsY,s.NCellsZ,dose_coll);

println(to)
idxX = Int(ceil(s.NCellsX/2))
idxY = Int(ceil(s.NCellsY/2))   #idxY = floor(Int,s.NCellsY/s.d*(0.5*s.d + s.y0))
idxZ = Int(ceil(s.NCellsZ/2))

X = (s.xMid'.*ones(size(s.yMid)))
Y = (s.yMid'.*ones(size(s.xMid)))'
Z = (s.zMid'.*ones(size(s.yMid)))
XZ = (s.xMid'.*ones(size(s.zMid)))
ZX = (s.zMid'.*ones(size(s.xMid)))
YZ = (s.yMid'.*ones(size(s.zMid)))'


# write vtk file
# vtkfile = vtk_grid("output/dose_fullPn_nPN$(s.nPN)", s.xMid, s.yMid,s.zMid)
# vtkfile["dose"] = dose_DLR
# vtkfile["dose_normalized"] = dose_DLR./sum(dose_DLR) * sum(mu_e)
# vtkfile["dose_uncollided"] = dose_DLR .- dose_coll
# vtkfile["dose_collided"] = dose_coll
# outfiles = vtk_save(vtkfile)

#uncomment to write out solution files
# save("output/dose_fullPn_nPN$(s.nPN)_$(s.tracerFileName).jld2", "dose", dose_DLR./sum(dose_DLR[dose_DLR.>0]) * sum(mu_e))

# plot results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, dpi=200);
im1 = ax1.pcolormesh(YZ',Z',dose_DLR[idxX,:,:]',vmin=0,vmax=maximum(dose_DLR[idxX,:,:]),cmap="jet")
plt.colorbar(im1,ax=ax1)
im2 = ax2.pcolormesh(YZ',Z',dose_coll[idxX,:,:]',vmax=maximum(dose_coll[idxX,:,:]),cmap="jet")
plt.colorbar(im2,ax=ax2)
im3 = ax3.pcolormesh(YZ',Z',dose_DLR[idxX,:,:]'.-dose_coll[idxX,:,:]',vmin=0,vmax=maximum(dose_DLR[idxX,:,:].-dose_coll[idxX,:,:]),cmap="jet")
plt.colorbar(im3,ax=ax3)
ax4.plot(s.zMid,dose_DLR[idxX,idxY,:],label="total dose")
ax4.plot(s.zMid,dose_coll[idxX,idxY,:],label="collided")
ax4.plot(s.zMid,(dose_DLR[idxX,idxY,:] .- dose_coll[idxX,idxY,:])+(dose_DLR[idxX,idxY+1,:] .- dose_coll[idxX,idxY+1,:]),label="uncollided")
ax1.title.set_text("Total dose")
ax2.title.set_text("Collided dose")
ax3.title.set_text("Uncollided dose")
ax4.title.set_text("Dose cut along z-axis")
ax4.legend()
ax1.axis("equal")
ax2.axis("equal")
ax3.axis("equal")
show()
tight_layout()
savefig("output/Results_fullPn_$(s.nPN)_$(s.tracerFileName).png")

println("main for config $(file_path) finished")
