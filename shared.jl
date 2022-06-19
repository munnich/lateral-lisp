using FFTW, WAV, Statistics, ArgParse, LinearAlgebra

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

"""
    readandgetfft(fnames::Array{String}, samplestart::Int, sampleend::Int)

Read and prepare input WAV file.
This cuts audio from samplestart until sampleend, subtracts mean, then 
normalizes.
Afterwards, the FFT is calculated and the absolute is saved.

The result is an array of `FFTResult` structs containing:
* `title::String`
* `waveform::Array{Float64}`
* `fs::Integer`
* `fftmagnitude::Array{Float64}`
"""
function readandgetfft(fnames::Array{String}, samplestart::Int, sampleend::Int)
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


