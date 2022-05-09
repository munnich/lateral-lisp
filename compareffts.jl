using Plots, FFTW, WAV, Statistics, ArgParse, LinearAlgebra

function parse()
    s = ArgParseSettings()
    @add_arg_table s begin
        "folder_a"
        help = "First data folder. Folder name will be re-used for titles."
        required = false
        arg_type = String
        default = "normal"
        "folder_b"
        help = "Second data folder. Folder name will be re-used for titles."
        required = false
        arg_type = String
        default = "lisp"
        "--format", "-f"
        help = "Graph output format."
        required = false
        arg_type = String
        default = "pdf"
        "--fft-start", "-s"
        help = "Select starting frequency for FFT plot."
        required = false
        arg_type = Int
        default = 1
        "--fft-end", "-e"
        help = "Select end frequency for FFT plot."
        required = false
        arg_type = Int
        default = 0
        "--mode", "-m"
        help = "Select analysis mode. Current options: end (default), slice."
        required = false
        arg_type = String
        default = "end"
        "--plot", "-p"
        help = "Enable plotting."
        action = :store_true
        "--sample-start", "-S"
        help = "Select starting sample to be analyzed in slice mode."
        required = false
        arg_type = Int
        default = 1
        "--sample-end", "-E"
        help = "Select end sample to be analyzed in slice mode."
        required = false
        arg_type = Int
        default = 0
    end
    
    parse_args(s)
end

# my prefered plotting backend
pyplot()

#= 
   FFT section

   This contains funcs to grab FFTs and other data from audio files
   along with a struct to house these.
=#

struct FFTResult
    title::String
    waveform::Array{Float64}
    fs::Integer
    fftmagnitude::Array{Float64}
end

# for reasons unbeknown to me this seems to use more memory than the array one
# I don't want to delete it until I truly understand how this is the case
function getfft(fname::String)
    title = replace(fname, "/" => ": ")
    title = replace(title, ".wav" => "")
    title = replace(title, "_" => " ")
    title = uppercasefirst(title)

    wavdata = wavread(fname)
    audio = wavdata[1]
    fs::Int = wavdata[2]
    # subtract mean
    audiomean = mean(audio)
    audio = audio .- audiomean
    normalize!(audio)

    fftdata = fft(audio)
    fftmagnitude = abs.(fftdata)

    FFTResult(title, audio, fs, fftmagnitude)
end

function getfft(fnames::Array{String}, samplestart::Int, sampleend::Int)
    numfiles = length(fnames)
    results = Array{FFTResult}(undef, numfiles)
    audio = Matrix{Float64}[]
    for i in 1:numfiles
        wavdata = wavread(fnames[i])

        wavaudio = wavdata[1]
        # subtract mean
        audiomean = mean(wavaudio)
        wavaudio = wavaudio .- audiomean
        # should obviously default to the full thing if unspecified
        if sampleend == 0
            sampleend = length(wavaudio)
        end
        # slicing the audio
        wavaudio = wavaudio[samplestart:1:sampleend, :]
        # it might be worth normalizing here?
        normalize!(wavaudio)
        # memory optimization - don't ask
        push!(audio, wavaudio)

        title = replace(fnames[i], "/" => ": ")
        title = replace(title, ".wav" => "")
        title = replace(title, "_" => " ")
        title = uppercasefirst(title)

        fftdata = fft(audio[i])

        #                      title  waveform  fs          fftmagnitude
        results[i] = FFTResult(title, audio[i], wavdata[2], abs.(fftdata))
    end
    return results
end

#=
   Slice examination and lisp detection section
=#

struct SliceResult
    title::String
    mean::Float64
    likely::Bool
    allmeans::Array{Float64}
end

function examineslice(input::Array{FFTResult}, slice::StepRange{Int64, Int64})
    # cut out a slice and compare FFTs
    # we use mean of the sliced normalized FFT magnitude to identify the lisp
    slicedfftmean = [mean(normalize(r.fftmagnitude)[slice]) for r in input]
    # ansatz: max mean in these frequencies is lisp!
    slicemax = findmax(slicedfftmean)
    # check if our max indeed has a mean in [0.0075, 0.0080] (magic number)
    checkthresh = slicemax[1] > 0.0075 && slicemax[1] < 0.0080

    return LispResult(input[slicemax[2]].title, slicemax[1], checkthresh, slicedfftmean)
end

# I'd prefer one result struct but slice and end print different info

struct EndResult
    title::String
    result::String
    diff::Float64
end

# this works reference-free
# since the original script here is for comparing this doesn't make sense
# hence this should probably be moved into its own script
function examineend(input::Array{FFTResult})
    output = Array{EndResult}(undef, length(input))
    for i in 1:length(input)
        # normalized bandpass with bounds 1000, (45% of fs) Hz
        # for <1 s samples upper bound is 90% highest freq
        slicedfft = normalize(input[i].fftmagnitude[1001:min(trunc(Int, input[i].fs * 0.45), trunc(Int, (end * 0.90)))])
        # we compare the last ≈ 1000 Hz of the above band to everything until then
        slicemeanend = mean(slicedfft[(end - 1000):end])
        slicemeanrest = mean(slicedfft[1:(end - 1000)])
        # mean(end) - mean(rest) > 0 => lisp
        slicemeandiff = slicemeanend - slicemeanrest
        if slicemeandiff > 0
            result = "lisp"
        else
            result = "normal"
        end
        output[i] = EndResult(input[i].title, result, slicemeandiff)
    end
    return output
end

#=
   Main section
=#

function main(args)
    println("Finished compilation, starting FFT examination...")
    # the filenames should be the same anyway so we only need to read folder_a
    folder_a = readdir(args["folder_a"])
    numfiles = length(folder_a)

    # arrays not tuples for theoretical support of >2 files
    results = Array{Array{FFTResult}}(undef, numfiles)
    lisps = Array{Union{SliceResult, Array{EndResult}}}(undef, numfiles)

    # the slice of frequencies we want to analyze
    # it feels ridiculous to hardcode this but that makes more sense to me
    slice = 1200:1:3500
    
    # multithreaded loop for getting and examining the ffts
    Threads.@threads for i in 1:numfiles
        results[i] = getfft(["$(args["folder_a"])/$(folder_a[i])",
                             "$(args["folder_b"])/$(folder_a[i])"],
                            args["sample-start"], args["sample-end"])
        if args["mode"] == "slice"
            lisps[i] = examineslice(results[i], slice)
        elseif args["mode"] == "end"
            lisps[i] = examineend(results[i])
        else
            error("Unknown mode chosen; options are slice and end")
        end
    end

    println("Preparing results...")
    # singlethreaded for plotting - this is the really inefficient part
    # printing needs to go here too
    for i in 1:numfiles
        # plotting is so slow it deserves a flag
        if args["plot"]
            plots = []
            for j in 1:length(results[i])
                # normal plot
                push!(plots, plot((1:1:length(results[i][j].waveform)) / results[i][j].fs,
                                  results[i][j].waveform, title=results[i][j].title,
                                  xlabel="Time [s]", ylabel="Amplitude"))
            end
            # why two loops? so we get the right order obviously!
            for j in 1:length(results[i])
                # defaulting
                if args["fft-end"] == 0
                    # waveform length affects max frequency
                    limit = results[i][j].fs#trunc(Int, min(results[i][j].fs / 2, length(results[i][j].waveform) / 2))
                else
                    limit = args["fft-end"]
                end
                # fft plot
                push!(plots, plot((args["fft-start"] - 1):(limit - 1),
                                  # a bit ridiculous maybe but >10 rarely happens so let's cut those off
                                  min.(10, results[i][j].fftmagnitude[args["fft-start"]:limit]),
                                  title="FFT " * results[i][j].title,
                                  xlabel="Frequency [Hz]", ylabel="Amplitude"))
            end
            plot(plots..., layout=(2, 2), legend=false, link=:y)
            figname = lowercase(folder_a[i])
            figname = replace(figname, ".wav" => "")
            savefig("output/$(args["folder_a"])_vs_$(args["folder_b"])_$figname.$(args["format"])")
        end

        if args["mode"] == "slice"
            println("$(lisps[i].title) detected as lisp with $(lisps[i].mean), " *
                    "within expected frequencies: $(lisps[i].likely), " *
                    "all means: $(lisps[i].allmeans)") 
        elseif args["mode"] == "end"
            for j in 1:2
                println("$(lisps[i][j].title) detected as $(lisps[i][j].result) " *
                        "with difference: $(lisps[i][j].diff)")
            end
        end
    end
    println("Finished.")
end

@time main(parse())
