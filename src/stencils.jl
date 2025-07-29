__precompile__

using SparseArrays

include("utils.jl")

function vectorIndex(nx, ny, i, j, k)
    return (k-1) * nx * ny + (j-1) * nx + i
end

function stencil_weights(x::AbstractVector{<:Real}, x₀::Real, m::Integer) #derivation: https://github.com/mitmath/matrixcalc/blob/iap2024/psets/pset2sol.pdf (problem 2d)
    ℓ = 0:length(x)-1
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    return A \ (ℓ .== m) # vector of weights w
end

struct UpwindStencil3DCUDA
    D⁺₁::CuSparseMatrixCSC{T, Int32}
    D⁺₂::CuSparseMatrixCSC{T, Int32}
    D⁺₃::CuSparseMatrixCSC{T, Int32}
    D⁻₁::CuSparseMatrixCSC{T, Int32}
    D⁻₂::CuSparseMatrixCSC{T, Int32}
    D⁻₃::CuSparseMatrixCSC{T, Int32}

    function UpwindStencil3DCUDA(settings::Settings, order::Int=2)
        nx, ny, nz = settings.NCellsX, settings.NCellsY, settings.NCellsZ
        Δx, Δy, Δz = settings.dx, settings.dy, settings.dz
        density = ones(size(settings.density,1),size(settings.density,2),size(settings.density,3))
        # density = settings.density 
        
        D⁺₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₃ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₃ = spzeros(nx*ny*nz, nx*ny*nz)

        if order == 1
            # First-order accuracy
            # Set up D⁺₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i-1, j, k)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δx / density[i, j, k]
                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δx / density[i-1, j, k]
                end
            end
            D⁺₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁺₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j-1, k)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δy / density[i, j, k]
                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δy / density[i, j-1, k]
                end
            end
            D⁺₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            # Set up D⁺₃
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j, k-1)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δz / density[i, j, k]
                if k > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δz / density[i, j, k-1]
                end
            end
            D⁺₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i+1,j,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δx / density[i,j,k]; 
                if i < nx
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δx / density[i+1,j,k]; 
                end
            end
            D⁻₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # y part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j+1,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δy / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δy / density[i,j+1,k]; 
                end
            end
            D⁻₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₃ 
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                 # z part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j,k+1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δz / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δz / density[i,j,k+1]; 
                end
            end
            D⁻₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
        elseif order == 2
            # Second-order accuracy
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₁
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i-1, j, k)
                indexMM = vectorIndex(nx, ny, i-2, j, k)

                II[counter] = index
                J[counter] = index
                vals[counter] = 3 / Δx / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = -4 / Δx / 2 / density[i-1, j, k]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = 1 / Δx / 2 / density[i-2, j, k]
            end
            D⁺₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₂
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3
                
                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j-1, k)
                indexMM = vectorIndex(nx, ny, i, j-2, k)

                II[counter] = index
                J[counter] = index
                vals[counter] = 3 / Δy / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = -4 / Δy / 2 / density[i, j-1, k]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = 1 / Δy / 2 / density[i, j-2, k]
            end
            D⁺₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₃
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j, k-1)
                indexMM = vectorIndex(nx, ny, i, j, k-2)

                II[counter] = index
                J[counter] = index
                vals[counter] = 3 / Δz / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = -4 / Δz / 2 / density[i, j, k-1]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = 1 / Δz / 2 / density[i, j, k-2]
            end
            D⁺₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₁
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i+1, j, k)
                indexMM = vectorIndex(nx, ny, i+2, j, k)

                II[counter] = index
                J[counter] = index
                vals[counter] = -3 / Δx / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = 4 / Δx / 2 / density[i+1, j, k]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = -1 / Δx / 2 / density[i+2, j, k]
            end
            D⁻₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₂ 
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j+1, k)
                indexMM = vectorIndex(nx, ny, i, j+2, k)

                II[counter] = index
                J[counter] = index
                vals[counter] = -3 / Δy / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = 4 / Δy / 2 / density[i, j+1, k]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = -1 / Δy / 2 / density[i, j+2, k]
            end
            D⁻₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₃ 
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j, k+1)
                indexMM = vectorIndex(nx, ny, i, j, k+2)

                II[counter] = index
                J[counter] = index
                vals[counter] = -3 / Δz / 2 / density[i, j, k]
                II[counter+1] = index
                J[counter+1] = indexM
                vals[counter+1] = 4 / Δz / 2 / density[i, j, k+1]
                II[counter+2] = index
                J[counter+2] = indexMM
                vals[counter+2] = -1 / Δz / 2 / density[i, j, k+2]
            end
            D⁻₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

        end

        new(D⁺₁, D⁺₂, D⁺₃, D⁻₁, D⁻₂, D⁻₃)
    end
end

struct UpwindStencil3DCUDA_nonuniform
    D⁺₁::CuSparseMatrixCSC{T, Int32}
    D⁺₂::CuSparseMatrixCSC{T, Int32}
    D⁺₃::CuSparseMatrixCSC{T, Int32}
    D⁻₁::CuSparseMatrixCSC{T, Int32}
    D⁻₂::CuSparseMatrixCSC{T, Int32}
    D⁻₃::CuSparseMatrixCSC{T, Int32}

    function UpwindStencil3DCUDA_nonuniform(settings::Settings, order::Int=2)
        nx, ny, nz = settings.NCellsX, settings.NCellsY, settings.NCellsZ
        Δx, Δy, Δz = settings.dx, settings.dy, settings.dz
        density = ones(size(settings.density,1),size(settings.density,2),size(settings.density,3))
        x_coords = settings.xMid
        y_coords = settings.yMid
        z_coords = settings.zMid

        # density = settings.density 
        
        D⁺₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₃ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₃ = spzeros(nx*ny*nz, nx*ny*nz)

        if order == 1
            # First-order accuracy
            # Set up D⁺₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i-1, j, k)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δx / density[i, j, k]
                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δx / density[i-1, j, k]
                end
            end
            D⁺₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁺₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j-1, k)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δy / density[i, j, k]
                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δy / density[i, j-1, k]
                end
            end
            D⁺₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            # Set up D⁺₃
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j, k-1)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δz / density[i, j, k]
                if k > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δz / density[i, j, k-1]
                end
            end
            D⁺₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i+1,j,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δx / density[i,j,k]; 
                if i < nx
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δx / density[i+1,j,k]; 
                end
            end
            D⁻₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # y part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j+1,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δy / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δy / density[i,j+1,k]; 
                end
            end
            D⁻₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            # Set up D⁻₃ 
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                Δz = 
                 # z part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j,k+1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δz / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δz / density[i,j,k+1]; 
                end
            end
            D⁻₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

        elseif order == 2
            # Second-order accuracy
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                index   = vectorIndex(nx, ny, i, j, k)
                indexM  = vectorIndex(nx, ny, i-1, j, k)
                indexMM = vectorIndex(nx, ny, i-2, j, k)

                x0 = x_coords[i]
                x1 = x_coords[i-1]
                x2 = x_coords[i-2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i-1, j, k]
                ρ2 = density[i-2, j, k]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end

            D⁺₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz))
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₂
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3
                
                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j-1, k)
                indexMM = vectorIndex(nx, ny, i, j-2, k)

                x0 = y_coords[j]
                x1 = y_coords[j-1]
                x2 = y_coords[j-2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i, j-1, k]
                ρ2 = density[i, j-2, k]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end
            D⁺₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₃
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j, k-1)
                indexMM = vectorIndex(nx, ny, i, j, k-2)

                x0 = z_coords[k]
                x1 = z_coords[k-1]
                x2 = z_coords[k-2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i, j, k-1]
                ρ2 = density[i, j, k-2]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end
            D⁺₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₁
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i+1, j, k)
                indexMM = vectorIndex(nx, ny, i+2, j, k)

                x0 = x_coords[i]
                x1 = x_coords[i+1]
                x2 = x_coords[i+2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i+1, j, k]
                ρ2 = density[i+2, j, k]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end
            D⁻₁ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));
            
            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₂ 
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j+1, k)
                indexMM = vectorIndex(nx, ny, i, j+2, k)

                
                x0 = y_coords[j]
                x1 = y_coords[j+1]
                x2 = y_coords[j+2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i, j+1, k]
                ρ2 = density[i, j+2, k]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end
            D⁻₂ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₃ 
            for i in 3:nx-2, j in 3:ny-2, k in 3:nz-2
                counter += 3

                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexM = vectorIndex(nx, ny, i, j, k+1)
                indexMM = vectorIndex(nx, ny, i, j, k+2)

                x0 = z_coords[k]
                x1 = z_coords[k+1]
                x2 = z_coords[k+2]

                c0, c1, c2 = stencil_weights([x0, x1, x2], x0, 1)

                ρ0 = density[i, j, k]
                ρ1 = density[i, j, k+1]
                ρ2 = density[i, j, k+2]

                II[counter]   = index
                J[counter]    = index
                vals[counter] = c0 / ρ0

                II[counter+1]   = index
                J[counter+1]    = indexM
                vals[counter+1] = c1 / ρ1

                II[counter+2]   = index
                J[counter+2]    = indexMM
                vals[counter+2] = c2 / ρ2
            end
            D⁻₃ = CuSparseMatrixCSC(sparse(II, J, T.(vals), nx*ny*nz, nx*ny*nz));

        end

        new(D⁺₁, D⁺₂, D⁺₃, D⁻₁, D⁻₂, D⁻₃)
    end
end