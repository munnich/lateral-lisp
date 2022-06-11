using FFTW, WAV, Statistics, ArgParse, LinearAlgebra


function parse()
    s = ArgParseSettings(description="Identify lisps in WAV recordings.")
    @add_arg_table s begin
        "folder"
        help = "Folder containing to-be-analyzed files."
        required = false
        arg_type = String
        default = "texts"
        "--lisp-start", "-l"
        help = "Lisp's starting frequency."
        required = false
        arg_type = Int
        default = 3000
        "--lisp-end", "-L"
        help = "Lisp's ending frequency."
        required = false
        arg_type = Int
        default = 4000
        "--normal-start", "-n"
        help = "Normal's starting frequency."
        required = false
        arg_type = Int
        default = 2500
        "--normal-end", "-N"
        help = "Normal's ending frequency."
        required = false
        arg_type = Int
        default = 3000
        "--rest-start", "-r"
        help = "Rest's starting frequency."
        required = false
        arg_type = Int
        default = 1000
        "--rest-end", "-R"
        help = "Rest's ending frequency."
        required = false
        arg_type = Int
        default = 4000
        "--segment-length", "-s"
        help = "Segment length in seconds."
        required = false
        arg_type = Float64
        default = 0.5
    end
    
    parse_args(s)
end

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
    getfft(fname::String, wavdata::Tuple)

Prepare audio and get FFT.
The wavdata input should be the result from `WAV`'s `wavread`.
Afterwards, the FFT is calculated and the absolute is saved.

The result is an `FFTResult` struct containing:
* `title::String`
* `waveform::Array{Float64}`
* `fs::Integer`
* `fftmagnitude::Array{Float64}`
"""
function getfft(fname::String, wavdata::Tuple)
    audio = wavdata[1]
    # subtract mean
    audiomean = mean(audio)
    audio = audio .- audiomean

    # it might be worth normalizing here?
    normalize!(audio)

    title = replace(fname, "/" => ": ")
    title = replace(title, ".wav" => "")
    title = replace(title, "_" => " ")
    title = uppercasefirst(title)

    fftdata = fft(audio)

    #                title  waveform  fs          fftmagnitude
    return FFTResult(title, audio, wavdata[2], abs.(fftdata))
end

struct SegmentResult
    title::String
    result::Bool
    diff::Float64
end

"""
    examinesegment(input::FFTResult, segment::Tuple{Int, Int}, rest::Tuple{Int, Int})

A reference-free examination algorithm.
It takes an FFTResult and performs a bandpass over it, then normalizes that.
It compares the amplitudes of the specified frequencies in the band to the rest of
the band by subtraction. If the result is greater zero, a lisp is assumed.

The result is an SegmentResult containing:
* `title::String`
* `result::Bool`
* `diff::Float64`
"""
function examinesegment(input::FFTResult, segment::Tuple{Int, Int}, rest::Tuple{Int, Int})
    # adjustment for differing sample lengths
    factor = length(input.waveform) / input.fs
    low = trunc(Int, rest[1] * factor)
    high = trunc(Int, rest[2] * factor)
    # normalized bandpass
    slicedfft = normalize(input.fftmagnitude[low:high])
    # scale the segment
    scaledsegment = [trunc(Int, (s - rest[1]) * factor) for s in segment]
    # we have to make sure the rest segment isn't greater than the whole length
    slicemean = mean(slicedfft[scaledsegment[1]:min(scaledsegment[2], end)])
    if segment[2] == length(slicedfft)
        slicemeanrest = mean(slicedfft[1:scaledsegment[1]])
    elseif segment[1] == 1
        slicemeanrest = mean(slicedfft[scaledsegment[2]:end])
    else
        slicemeanrest = mean(vcat(slicedfft[1:scaledsegment[1]],
                                  slicedfft[scaledsegment[2]:end]))
    end
    # mean(end) - mean(rest) > 0 ⇒ lisp
    slicemeandiff = slicemean - slicemeanrest
    result = slicemeandiff > 0
    return SegmentResult(input.title, result, slicemeandiff)
end

struct TextResult
    title::String
    result::Bool
    hits::Int
    misses::Int
end

"""
    examinetext(file::String, lisp::Tuple{Int, Int}, normal::Tuple{Int, Int},
                rest::Tuple{Int, Int}, seglength::Float64)

Examine text files for lisps based on examinesegment.
Just counts the number of lisps vs. non-lisps detected and compares.

The result is a TextResult containing:
* `title::String`
* `result::Bool`
* `hits::Int`
* `misses::Int`
"""
function examinetext(file::String, lisp::Tuple{Int, Int}, normal::Tuple{Int, Int},
        rest::Tuple{Int, Int}, seglength::Float64)
    wavdata = wavread(file)
    audio = wavdata[1]
    fs = wavdata[2]

    audiomean = mean(abs.(audio))

    # determine segment length based on sampling frequency
    segmentlength = trunc(Int, fs * seglength)

    # filter silent segments
    segments = []
    for i in 1:segmentlength:(length(audio) - segmentlength)
        slice = audio[i:(i + segmentlength)]
        # all segments with mean under audiomean are assumed silent
        if mean(abs.(slice)) > audiomean
            push!(segments, getfft(file, (slice, fs)))
        end
    end

    # run through the segments and count the detected lisps and non-lisps
    hits = 0
    misses = 0
    for segment in segments
        # [3000, 4000] Hz seems to be the lisp
        if examinesegment(segment, lisp, rest).result
            hits += 1
        # [2500, 3000] Hz works well enough for the normal s
        elseif examinesegment(segment, normal, rest).result
            misses += 1
        end
    end

    # return true if lisp is detected off the assumption that
    # hits > misses ⇒ lisp ∧ hits < misses ⇒ ¬lisp
    return TextResult(file, hits > misses, hits, misses)
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
    folder = readdir(args["folder"])
    numfiles = length(folder)

    results = Array{TextResult}(undef, numfiles)

    # multithreaded loop for getting and examining the ffts
    # unfortunately the stuff we can use multithreading for isn't that slow
    Threads.@threads for i in 1:numfiles
        results[i] = examinetext("$(args["folder"])/$(folder[i])",
                                 (args["lisp-start"], args["lisp-end"]),
                                 (args["normal-start"], args["normal-end"]),
                                 (args["rest-start"], args["rest-end"]),
                                 args["segment-length"])
    end

    println("Finished analysis:")
    # singlethreaded for printing
    for i in 1:numfiles
        println("$(results[i].title) is lisp: $(results[i].result), " *
                "with $(results[i].hits) hits, $(results[i].misses) misses")
    end
    println("Finished.")
end

main(parse())
