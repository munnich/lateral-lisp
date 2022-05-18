using Plots, FFTW, WAV, Statistics, ArgParse, LinearAlgebra


function parse()
    s = ArgParseSettings(description="Identify lisps in WAV recordings.")
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

"""
    getfft(fnames::Array{String}, samplestart::Int, sampleend::Int)

Read and prepare input WAV file.
This cuts audio from samplestart until sampleend, subtracts mean, then 
normalizes.
Afterwards, the FFT is calculated and the absolute is saved.

The result is an FFTResult struct containing:
* `title::String`
* `waveform::Array{Float64}`
* `fs::Integer`
* `fftmagnitude::Array{Float64}`
"""
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

"""
    examineslice(input::Array{FFTResult}, slice::StepRange{Int64, Int64})

Examine a slice by comparing FFT magnitude spectrums to identify lisps.
Only one lisp is detected, so this algorithm just finds the recording most likely 
to be a lisp.
This is done by normalizing the slice and finding the larger mean, which is
assumed to be the lisp.
For reference, the values are compared to magic numbers that are based on one set
of samples and should be taken with a giant hand of salt.

The result is a SliceResult struct containing:
* `title::String`
* `mean::Float64`
* `likely::Bool`
* `allmeans::Array{Float64}`
"""
function examineslice(input::Array{FFTResult}, slice::StepRange{Int64, Int64})
    # cut out a slice and compare FFTs
    # we use mean of the sliced normalized FFT magnitude to identify the lisp
    slicedfftmean = [mean(normalize(r.fftmagnitude)[slice]) for r in input]
    # ansatz: max mean in these frequencies is lisp!
    slicemax = findmax(slicedfftmean)
    # check if our max indeed has a mean in [0.0075, 0.0080] (magic number)
    checkthresh = slicemax[1] > 0.0075 && slicemax[1] < 0.0080

    return SliceResult(input[slicemax[2]].title, slicemax[1], checkthresh, slicedfftmean)
end

# I'd prefer one result struct but slice and end print different info

struct EndResult
    title::String
    result::String
    diff::Float64
end

# since the original script here is for comparing this doesn't make sense
# hence this should probably be moved into its own script
"""
    examineend(input::FFTResult)

A reference-free examination algorithm.
It takes an FFTResult and performs a bandpass over it, then normalizes that.
It compares the amplitudes of the last frequencies in the band to the rest of
the band by subtraction. If the result is greater zero, a lisp is assumed.

The result is an EndResult containing:
* `title::String`
* `result::String`
* `diff::Float64`
"""
function examineend(input::FFTResult)
    # adjustment for differing sample lengths
    factor = length(input.waveform) / input.fs
    low = trunc(Int, 1001 * factor)
    high = trunc(Int, 4001 * factor)
    # normalized bandpass
    slicedfft = normalize(input.fftmagnitude[low:high])
    # we compare the last couple freqs of the above band to everything until then
    interest = trunc(Int, 1000 * factor)
    slicemeanend = mean(slicedfft[(end - interest):end])
    slicemeanrest = mean(slicedfft[1:(end - interest)])
    # mean(end) - mean(rest) > 0 ⇒ lisp
    slicemeandiff = slicemeanend - slicemeanrest
    if slicemeandiff > 0
        result = "lisp"
    else
        result = "normal"
    end
    return EndResult(input.title, result, slicemeandiff)
end

#=
   Full text analysis

   our first task after having figured out examineend seems to work well enough
   is to analyze a full text by counting the number of lisps vs normals

   concept:
   >load file
   >save mean
   >split into segments
    >overlapping?
   >if mean < full text mean: assume silence
   >test non-silent segments for lisp
   
   ideas:
   - overlapping segments?
   - smaller frequency band for analysis
=#

function examinetext(file::String)
    wavdata = wavread(file)
    audio = wavdata[1]
    fs = wavdata[2]

    audiomean = mean(audio)

    # length of 0.5 s => segment length is fs / 2 s
    segmentlength = fs / 2

    # filter silent segments
    segments = []
    for i in 1:segmentlength:(length(audio) - segmentlength)
        slice = audio[i:(i + segmentlength)]
        if mean(slice) > audiomean
            segments = cat(1, segments, slice)
        end
    end

    println("$file: $(length(segments)) / $(length(audio)) non-silent")
end

#=
   Main section
=#

"""
The main function. Runs the functions with multithreading and prints results
single-threaded with optional plotting.
"""
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
    # unfortunately the stuff we can use multithreading for isn't that slow
    Threads.@threads for i in 1:numfiles
        results[i] = getfft(["$(args["folder_a"])/$(folder_a[i])",
                             "$(args["folder_b"])/$(folder_a[i])"],
                            args["sample-start"], args["sample-end"])
        if args["mode"] == "slice"
            lisps[i] = examineslice(results[i], slice)
        elseif args["mode"] == "end"
            # the "end" algorithm runs on single files at a time
            # since we're comparing files we do the grouping here
            output = Array{EndResult}(undef, length(results[i]))
            for j in 1:length(results[i])
                output[j] = examineend(results[i][j])
            end
            lisps[i] = output
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
            # create the output directory if it doesn't exist yet
            mkpath("output")
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
                    # untested - but you can just manually set the fft-end
                    limit = results[i][j].fs ^ 2 / length(results[i][j].waveform)
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
            # this combines all the above plots with 2x2 subplots
            # also links the y-axis horizontally
            # we only plot one line per subplot so legends are nonsense
            plot(plots..., layout=(2, 2), legend=false, link=:y)
            # prepare the output filename
            # with defaults and title "foo_1" we write to "output/normal_vs_lisp_foo_1.pdf"
            figname = lowercase(folder_a[i])
            figname = replace(figname, ".wav" => "")
            savefig("output/$(args["folder_a"])_vs_$(args["folder_b"])_$figname.$(args["format"])")
        end

        # the lisp results differ for the two modes so they need their own prints
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

main(parse())
