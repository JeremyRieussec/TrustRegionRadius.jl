# Save the algorithm information, parameters, and problem name to a text file
function save_algorithm_info_txt(algo_info::AlgorithmInfo,  filename::String)
    open(filename, "w") do io
        println(io, algo_info)
        println(io, "----------------------------------------------------------------------")
        println(io, "Optimization Path (accepted steps):")
        for (i, x) in enumerate(algo_info.path)
            println(io, "Step $(i-1): $x")
        end
    end
    println("Algorithm information and parameters saved to $filename")
end