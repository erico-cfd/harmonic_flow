#------------------------------------------------------------------------
# Author: Erico Lopes de Souza
# Université Paris Saclay / Universidade de Brasilia (UnB)
# Script to plot mean velocity amplitude and mean phase angle
# for a hemi-equilateral triangular duct under an oscillatory
# pressure gradient (Womersley flow).
#------------------------------------------------------------------------
 
using Plots 
using LaTeXStrings

const A = sqrt(3) / 6
const N = 60

λ_mn(m, n) = 4π^2 / 3 * (3m^2 + 3m*n + n^2)

function I_phi(m, n)
    t1 = (1 - (-1)^m) / (m  * (3m + 2n))
    t2 = (1 - (-1)^n) / (n  * (2m +  n))
    t3 = (1 - (-1)^(m+n)) / ((m+n) * (3m + n))
    return sqrt(3) / π^2 * (t1 + t2 - t3)
end

const I_phi2 = sqrt(3) / 8

function wbar_cs(α)
    wc = 0.0
    ws = 0.0
    α4 = α^4
    for m in 1:N, n in 1:N
        lmn = λ_mn(m, n)
        Imn = I_phi(m, n)
        iszero(Imn) && continue #parity filter
        fac = Imn^2 / I_phi2
        den = lmn^2 + α4
        wc += lmn  / den * fac
        ws += α^2  / den * fac
    end
    return wc / A, ws / A
end

alphas = range(0.0, 100.0; length=600)

wa_vals    = Vector{Float64}(undef, length(alphas))
theta_vals = Vector{Float64}(undef, length(alphas))

for (i, α) in enumerate(alphas)
    wc, ws         = wbar_cs(α)
    wa_vals[i]     = hypot(wc, ws)          # eq. (29): w̄_a = √(w̄_c² + w̄_s²)
    theta_vals[i]  = -atan(ws, wc)          # eq. (29): θ̄ = -atan2(w̄_s, w̄_c)
end

println("w̄_a at α = 0   : $(wa_vals[1])")
println("θ̄   at α = 100 : $(theta_vals[end]) rad  ($(round(theta_vals[end]*180/π, digits=2))°)")
println("θ̄   limit (α→∞): should approach -π/2 ≈ $(-π/2)")

p1 = plot(alphas, wa_vals;
    xlabel     = L"\alpha",
    ylabel     = L"\bar{w}_a",
    title      = "Mean velocity amplitude vs "*L"\alpha",
    lw         = 2,
    color      = :blue,
    legend     = false,
    ylims      = (0, Inf),
    xlims      = (0, 100),
    framestyle = :box,
    grid       = true,
)

p2 = plot(alphas, theta_vals;
    xlabel     = L"\alpha",
    ylabel     = L"\bar{\theta}",
    title      = "Mean phase angle vs "*L"\alpha",
    lw         = 2,
    color      = :red,
    legend     = false,
    yticks     = ([-π/2, -3π/8, -π/4, -π/8, 0.0],
                  [L"-\pi/2", L"-3\pi/8", L"-\pi/4", L"-\pi/8", "0"]),
    ylims      = (-π/2 - 0.05, 0.05),
    xlims      = (0, 100),
    framestyle = :box,
    grid       = true,
)

savefig(p1, joinpath(@__DIR__, "wbar_amplitude.png"))
savefig(p2, joinpath(@__DIR__, "theta_bar.png"))

combined = plot(p1, p2; layout=(1, 2), size=(1000, 420), margin=5Plots.mm)
savefig(combined, joinpath(@__DIR__, "womersley_validation.png"))

display(combined)
println("\nGraphs saved: wbar_amplitude.png, theta_bar.png, womersley_validation.png")
