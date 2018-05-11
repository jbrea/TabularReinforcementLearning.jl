function walkenv(callback)
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "environments"))
    for file in files
        (name, ext) = splitext(file)
        if ext == ".jl"
            callback(root, file, name)
        end
    end
end
end

function listenvironments()
list = []
walkenv((root, file, name) -> push!(list, name))
list
end
function loadenvironment(envname)
walkenv((root, file, name) -> if name == envname 
            include(joinpath(root, file))
        end)
end

export listenvironments, loadenvironment

