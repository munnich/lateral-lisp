using Plots, FFTW, WAV, Statistics, ArgParse

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
		"--start", "-s"
		help = "Select starting frequency for FFT plot."
		required = false
		arg_type = Int
		default = 1
		"--end", "-e"
		help = "Select end frequency for FFT plot."
		required = false
		arg_type = Int
		default = 0
	end

	parse_args(s)
end

pyplot()

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
    title = uppercasefirst(title)

    wavdata = wavread(fname)
    audio = wavdata[1]
    fs::Int = wavdata[2]
    # subtract mean to account for volume differences
    audiomean = mean(audio)
    audio = audio .- audiomean

    fftdata = fft(audio)
    fftmagnitude = abs.(fftdata)

    FFTResult(title, audio, fs, fftmagnitude)
end

function getfft(fnames::Array{String})
    numfiles = length(fnames)
    results = Array{FFTResult}(undef, numfiles)
	audio = Matrix{Float64}[]
	for i in 1:numfiles
        wavdata = wavread(fnames[i])

		wavaudio = wavdata[1]
		# subtract mean to account for volume differences
		audiomean = mean(wavaudio)
		wavaudio = wavaudio .- audiomean
        # memory optimization - don't ask
        push!(audio, wavaudio)

		title = replace(fnames[i], "/" => ": ")
		title = replace(title, ".wav" => "")
		title = uppercasefirst(title)

        fftdata = fft(audio[i])

        #                      title  waveform  fs          fftmagnitude
        results[i] = FFTResult(title, audio[i], wavdata[2], abs.(fftdata))
	end
    return results
end

function main(args)
	numfiles = length(readdir(args["folder_a"]))

    # arrays not tuples for theoretical support of >2 files
    results = Array{Array{FFTResult}}(undef, numfiles)
	
    # multithreaded loop for getting the ffts
	Threads.@threads for i in 1:numfiles
        results[i] = getfft(["$(args["folder_a"])/$i.wav", "$(args["folder_b"])/$i.wav"])
	end

    # singlethreaded for plotting - this is the really inefficient part
    for i in 1:numfiles
        plots = []
        for j in 1:length(results[i])
            # normal plot
            push!(plots, plot((1:1:length(results[i][j].waveform)) / results[i][j].fs,
                              results[i][j].waveform, title=results[i][j].title,
                              xlabel="Time [s]", ylabel="Amplitude"))
            # defaulting
            if args["end"] == 0
                limit = results[i][j].fs
            else
                limit = args["end"]
            end
            # fft plot
            push!(plots, plot((args["start"] - 1):(limit - 1),
                              results[i][j].fftmagnitude[args["start"]:limit],
                              title="FFT" * results[i][j].title,
                              xlabel="Frequency [Hz]", ylabel="Amplitude"))
        end
        plot(plots..., layout=(2, 2), legend=false)
		savefig("output/$(args["folder_a"])_vs_$(args["folder_b"])_$i.$(args["format"])")

        # cut out a slice and compare FFTs
        # [1200, 3500] seems to work quite well
        # lisp's mean in this range is usually in [1.5, 1.6]
        slice = 1200:1:3500
        slicedfftmean = [mean(r.fftmagnitude) for r in results[i]]
        slicemax = findmax(slicedfftmean)
        println("$(results[i][slicemax[2]].title) detected as lisp with $(slicemax[1]), " *
                "all means: $slicedfftmean") 
    end
end

@time main(parse())
