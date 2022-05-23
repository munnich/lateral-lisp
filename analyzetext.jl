using FFTW, WAV, Statistics, ArgParse, LinearAlgebra


function parse()
    s = ArgParseSettings(description="Identify lisps in WAV recordings.")
    @add_arg_table s begin
        "folder"
        help = "Folder containing to-be-analyzed files."
        required = false
        arg_type = String
        default = "texts"
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
    getfft(fname::String, wavdata::Tuple, samplestart::Int, sampleend::Int)

Prepare audio and get FFT.
The wavdata input should be the result from `WAV`'s `wavread`.
This cuts audio from samplestart until sampleend, subtracts mean, then 
normalizes.
Afterwards, the FFT is calculated and the absolute is saved.

The result is an `FFTResult` struct containing:
* `title::String`
* `waveform::Array{Float64}`
* `fs::Integer`
* `fftmagnitude::Array{Float64}`
"""
function getfft(fname::String, wavdata::Tuple, samplestart::Int, sampleend::Int)
    audio = wavdata[1]
    # subtract mean
    audiomean = mean(audio)
    audio = audio .- audiomean
    # should obviously default to the full thing if unspecified
    if sampleend == 0
        sampleend = length(audio)
    end
    # slicing the audio
    audio = audio[samplestart:1:sampleend, :]
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

struct EndResult
    title::String
    result::String
    diff::Float64
end

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

struct TextResult
    title::String
    result::Bool
    hits::Int
    misses::Int
end

"""
    examinetext(file::String, samplestart::Int, sampleend::Int)

Examine text files for lisps based on examineend.
The result is a TextResult containing:
* `title::String`
* `result::Bool`
* `hits::Int`
* `misses::Int`
"""
function examinetext(file::String, samplestart::Int, sampleend::Int)
    wavdata = wavread(file)
    audio = wavdata[1]
    fs = wavdata[2]

    audiomean = mean(audio)

    # length of 0.5 s => segment length is fs / 2 s
    segmentlength = fs / 2

    # filter silent segments
    segments = Array{FFTResult}
    for i in 1:segmentlength:(length(audio) - segmentlength)
        slice = audio[i:(i + segmentlength)]
        # all segments with mean under audiomean are assumed silent
        if mean(slice) > audiomean
            segments = cat(1, segments,
                           getfft(fname, (slice, fs), samplestart, sampleend))
        end
    end

    # verbosity - this should be removed if the current algorithm works
    println("$file: $(length(segments)) / $(length(audio)) non-silent")

    #=
    # this is where this no longer makes much sense - we need to identify
    # whether the segment should even have a lispable sound
    # need to do research into how to do this
    #
    # possible methods:
    # >formants
    # >use the same method as examineend
    # >fit some distribution to FFT magnitude of S and SL
    # >worst case scenario: machine learning
    =#

    # run through the segments and count the detected lisps and non-lisps
    hits = 0
    misses = 0
    for segment in segments
        num += Int(examineend(segment).result) * 2 - 1
    end

    # verbosity - this should be removed if the current algorithm works
    println("$file: $hits lisps, $misses non-lisps")

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

    # the slice of frequencies we want to analyze
    # it feels ridiculous to hardcode this but that makes more sense to me
    slice = 1200:1:3500
    
    # multithreaded loop for getting and examining the ffts
    # unfortunately the stuff we can use multithreading for isn't that slow
    Threads.@threads for i in 1:numfiles
        results[i] = examinetext(results[i][j])
    end

    println("Finished analysis:")
    # singlethreaded for printing
    for i in 1:numfiles
        println("$(results[i].title) is lisp: $(results[i].result) " *
                "with $(results[i].hits) hits, $(results[i].misses) misses")
    end
    println("Finished.")
end

main(parse())