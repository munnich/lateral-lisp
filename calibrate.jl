using FFTW, WAV, Statistics, ArgParse, LinearAlgebra
using Plots

pyplot()
include("shared.jl")

function parse()
    s = ArgParseSettings(description="Calibration helper for lisp detection.")
    @add_arg_table s begin
        "normal"
        help = "Normal pronunciation's audio file for calibration."
        required = true
        arg_type = String
        "lisp"
        help = "Lisp's audio file for calibration."
        required = true
        arg_type = String
        "--output", "-o"
        help = "Output file path"
        required = false
        arg_type = String
        default = "calibration.pdf"
        "--highpass", "-p"
        help = "High pass boundary frequency."
        required = false
        arg_type = Int
        default = 1000
        "--sample-start", "-S"
        help = "Select starting sample to be analyzed in slice mode."
        required = false
        arg_type = Int
        default = 1
        "--sample-end", "-E"
        help = "Select end sample to be analyzed in slice mode."
        required = false
        arg_type = Int
        default = 8000
    end
    
    parse_args(s)
end


# https://github.com/tungli/Findpeaks.jl
"""
`findpeaks(y::Array{T},
x::Array{S}=collect(1:length(y))
;min_height::T=minimum(y), min_prom::T=minimum(y),
min_dist::S=0, threshold::T=0 ) where {T<:Real,S}`\n
Returns indices of local maxima (sorted from highest peaks to lowest)
in 1D array of real numbers. Similar to MATLAB's findpeaks().\n
*Arguments*:\n
`y` -- data\n
*Optional*:\n
`x` -- x-data\n
*Keyword*:\n
`min_height` -- minimal peak height\n
`min_prom` -- minimal peak prominence\n
`min_dist` -- minimal peak distance (keeping highest peaks)\n
`threshold` -- minimal difference (absolute value) between
 peak and neighboring points\n
"""
function findpeaks(
                   y :: AbstractVector{T},
                   x :: AbstractVector{S} = collect(1:length(y))
                   ;
                   min_height :: T = minimum(y),
                   min_prom :: T = zero(y[1]),
                   min_dist :: S = zero(x[1]),
                   max_dist :: S = zero(x[1]),
                   threshold :: T = zero(y[1]),
                  ) where {T <: Real, S}

    dy = diff(y)

    peaks = in_threshold(dy, threshold)

    yP = y[peaks]
    peaks = with_prominence(y, peaks, min_prom)
    
    #minimal height refinement
    peaks = peaks[y[peaks] .> min_height]
    yP = y[peaks]

    peaks = with_distance(peaks, x, y, min_dist)

    peaks = within_distance(peaks, x, y, max_dist)

    peaks
end

"""
Select peaks that are inside threshold.
"""
function in_threshold(
                      dy :: AbstractVector{T},
                      threshold :: T,
                     ) where {T <: Real}

    peaks = 1:length(dy) |> collect

    k = 0
    for i = 2:length(dy)
        if dy[i] <= -threshold && dy[i-1] >= threshold
            k += 1
            peaks[k] = i
        end
    end
    peaks[1:k]
end

"""
Select peaks that have a given prominence
"""
function with_prominence(
                         y :: AbstractVector{T},
                         peaks :: AbstractVector{Int},
                         min_prom::T,
                        ) where {T <: Real}

    #minimal prominence refinement
    peaks[prominence(y, peaks) .> min_prom]
end


"""
Calculate peaks' prominences
"""
function prominence(y::AbstractVector{T}, peaks::AbstractVector{Int}) where {T <: Real}
    yP = y[peaks]
    proms = zero(yP)

    for (i, p) in enumerate(peaks)
        lP, rP = 1, length(y)
        for j = (i-1):-1:1
            if yP[j] > yP[i]
                lP = peaks[j]
                break
            end
        end
        ml = minimum(y[lP:p])
        for j = (i+1):length(yP)
            if yP[j] > yP[i]
                rP = peaks[j]
                break
            end
        end
        mr = minimum(y[p:rP])
        ref = max(mr,ml)
        proms[i] = yP[i] - ref
    end

    proms
end

"""
Select only peaks that are further apart than `min_dist`
"""
function with_distance(
                       peaks :: AbstractVector{Int},
                       x :: AbstractVector{S},
                       y :: AbstractVector{T},
                       min_dist::S,
                      ) where {T <: Real, S}

    peaks2del = zeros(Bool, length(peaks))
    inds = sortperm(y[peaks], rev=true)
    permute!(peaks, inds)
    for i = 1:length(peaks)
        for j = 1:(i-1)
            if abs(x[peaks[i]] - x[peaks[j]]) <= min_dist
                if !peaks2del[j]
                    peaks2del[i] = true
                end
            end
        end
    end

    peaks[.!peaks2del]
end

"""
Select only peaks that are closer together than `max_dist`
"""
function within_distance(
                       peaks :: AbstractVector{Int},
                       x :: AbstractVector{S},
                       y :: AbstractVector{T},
                       max_dist::S,
                      ) where {T <: Real, S}

    peaks2del = zeros(Bool, length(peaks))
    inds = sortperm(y[peaks], rev=true)
    permute!(peaks, inds)
    for i = 1:length(peaks)
        for j = 1:(i-1)
            if abs(x[peaks[i]] - x[peaks[j]]) >= max_dist
                if !peaks2del[j]
                    peaks2del[i] = true
                end
            end
        end
    end

    peaks[.!peaks2del]
end

function main(args)
    # read the files
    data = readandgetfft([args["normal"], args["lisp"]], args["sample-start"], args["sample-end"])
    start = args["highpass"]
    # set a range from high pass boundary to highest frequency
    range = start:trunc(Int, data[1].fs / 2)
    plots = []
    # loop over both files and run calibration
    for i in 1:2
        # prepare the calibration data of the fft magnitude
        ofinterest = data[i].fftmagnitude[range]
        # plot the fft magnitude
        x = (range |> collect)
        p = plot(title=data[i].title)
        plot!(p, data[i].fftmagnitude[1:trunc(Int, data[2].fs / 2)], label="FFT data")
        # find the peaks with max distance of 1000 and height max(fftmagnitude) / 4
        peaks = findpeaks(ofinterest, x, max_dist=1000, min_height=maximum(ofinterest) / 4, min_prom=0.0, min_dist=0, threshold=0.0)
        # the peaks have to be adjusted by the high pass boundary
        peakrange = [minimum(peaks), maximum(peaks)] .+ start
        # print and plot the range
        println("Detected peak range: ", peakrange)
        vspan!(p, peakrange, alpha=0.5, label="Peak range")
        push!(plots, p)
    end
    # plot the full thing in a smaller side-by-side plot
    plot(plots..., legend=false, xlabel="Frequency [Hz]", ylabel="Amplitude", size=(600, 200))
    savefig(args["output"])
end

main(parse())
