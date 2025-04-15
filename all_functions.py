# -*- coding: utf-8 -*-
"""
Simulation code for "On the Optimal Performance of Distributed Cell-Free Massive MIMO with LoS Propagation", IEEE Wireless Communications and Networking Conference (WCNC), March 2025
Author: Noor Ul Ain

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file

This file has all the functions required to simulate an uplink transmission in (both centralized and distributed) cellfree massive MIMO networks in the presence of los propagation.
"""

import copy
import numpy as np
import cmath
import os
from scipy import integrate, linalg
from math import exp, pi, sin, cos, sqrt, radians


def simulation_area(params_dict):
    """
    Functions to generate one network scenario with random placement of APs and UEs.
    For each UE and AP pair, it returns k-factor, channel gain, spatial correlation, channel mean
    Input:
        network parameters
    Output:
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE with kappa component
        gainOverNoise_B: LxK, Channel gain between each UE and AP
        H_Mean: LxKxN LOS component with kappa component
        kappa : LxK, los strength k-factor between each UE and AP
    """

    area = params_dict["area"]
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    bandwidth = params_dict["bandwidth"]
    fc = params_dict["carrier_frequency"]
    h = params_dict["delta_h"]  # 11m
    antennaSpacing = 1 / 2  # half wavelength
    sigma_sf_los = 8  # dB
    noiseVariancedB = -174 + pow2db(bandwidth) + 7  # Compute noise power (in dBm)

    # Initializations to store values
    Beta = np.zeros((L, K))
    R = np.zeros((L, K, N, N), dtype=complex)
    gainOverNoise_B = np.zeros((L, K))
    H_Mean = np.zeros((L, K, N), dtype=complex)
    kappa = np.zeros((L, K))
    dist_UEs_to_APs = np.zeros((L, K))

    # Deploying APs and UEs randomly within the service area
    UEpositions = (np.random.rand(K, 1) + 1j * np.random.rand(K, 1)) * area
    APpositions = (np.random.rand(L, 1) + 1j * np.random.rand(L, 1)) * area

    # wraping-around to mimic infinite area
    wrapHorizontal = np.tile([-area, 0, area], (3, 1))
    wrapVertical = wrapHorizontal.T
    wrapLocations = wrapHorizontal.T.flatten() + 1j * wrapVertical.T.flatten()
    APpositionsWrapped = np.tile(APpositions, [1, len(wrapLocations)]) + np.tile(wrapLocations, [L, 1])

    # Calculating the shortest distance of UE from each AP and finding nearest AP
    for k in range(0, K):
        dist_each_UE_to_APs = abs(APpositionsWrapped - np.tile(UEpositions[k, :], (L, 9))).min(axis=1)
        nearest_AP = abs(APpositionsWrapped - np.tile(UEpositions[k, :], (L, 9))).argmin(axis=1)
        dist_UEs_to_APs[:, k] = np.sqrt(h ** 2 + dist_each_UE_to_APs ** 2)

    # Calculated rician factor using equation 12
    kappa[:] = db2pow(13 - 0.03 * dist_UEs_to_APs[:])

    for k in range(0, K):
        # step 4: Calculate gain channel using equation 4
        Beta[:, k] = 35.4 - 20 * np.log10(fc) - 26 * np.log10(dist_UEs_to_APs[:, k])  # lx1

        gainOverNoise_B[:, k] = Beta[:, k] - noiseVariancedB + (
                np.random.randn(L) * sigma_sf_los)  # In power:gain*shadowing/noise

        # Calculate spatial correlation matrix R
        for l in range(0, L):
            theta = np.arcsin(h / dist_UEs_to_APs[l, k])
            phi = np.angle(UEpositions[k] - APpositionsWrapped[l, nearest_AP[l]], deg=False)[0]

            R[l, k, :, :] = db2pow(gainOverNoise_B[l, k]) * calculate_local_scattering(N, phi, theta,
                                                                                       antennaSpacing)
            # calculating channel mean (LOS component) from equation 41
            H_Mean[l, k, :] = sqrt(db2pow(gainOverNoise_B[l, k])) * np.exp(
                1j * 2 * pi * antennaSpacing * np.arange(0, N) * sin(phi) * cos(theta))
    return R, gainOverNoise_B, H_Mean, kappa


def initial_access(params_dict, Beta):
    """
    Function for pilot assignment to UEs and cooperation cluster formation using algorithm in section 5.2 (step 6)
    Input:
        Beta: LxK  Channel gains between L APs and K UEs
    Output:
        pilot_indicies_: Kx1: list consisting of index of assigned pilot to each UE
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        D_cellular: LxK, D matrix for cellular/smallcell. Each UE will be served by one AP
    """
    L = params_dict["L"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]

    D_cellular = np.zeros((L, K))
    pilot_indicies_ = -1 * np.ones((K, 1))
    for k in range(0, K):
        main_AP_index = Beta[:, k].argmax()  # master AP with the highest channel gain for UE k
        D_cellular[main_AP_index, k] = 1  # In cellular, only one AP with max channel gain is serving UE k
        pilot_indicies_ = pilot_assignment(k, main_AP_index, tau_p, pilot_indicies_, Beta)

    D_DCC = createDCC(D_cellular, L, tau_p, Beta, pilot_indicies_)
    return D_cellular, D_DCC, pilot_indicies_


def channel_generation(params_dict, R, H_Mean, kappa):
    # Function to generate los channels according to defination in section 5
    # Channel realizations are drawn from Gaussian distribution ~N_c(h_mean,R) to capture los and small scale fading.
    """
    Input:
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE
        H_Mean: LxKxN Rician LoS component (channel mean)
        kappa: LxK Rician factor between L APs and K UEs

    Output:
        H: LxKxNxrealizations, random channel realizations
        H_Mean: LxKxN LOS component with kappa component
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE with kappa component
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    realizations = params_dict["realizations"]

    H_Mean = np.tile(H_Mean[:, :, :, np.newaxis], (1, 1, 1, realizations))

    H = sqrt(0.5) * (np.random.randn(L, K, N, realizations) + 1j * np.random.randn(L, K, N, realizations))

    for l in range(0, L):
        for k in range(0, K):
            H_Mean[l, k, :, :] = sqrt(kappa[l, k] / (1 + kappa[l, k])) * H_Mean[
                                                                         l, k, :, :]
            R[l, k, :, :] = (1 / (1 + kappa[l, k])) * R[l, k, :, :]
            H[l, k, :, :] = H_Mean[l, k, :, :] + linalg.sqrtm(R[l, k, :, :]) @ H[l, k, :, :]
    return H, H_Mean, R


def channel_estimation(params_dict, R, pilot_indicies_, H, H_Mean, p_k):
    """ Function to calculate the channel estimates, see section 2B

    Input:
        R: LxKxNxN Normalized Spatial correlation matrix of NxN size between each lth AP and kth UE
        pilot_indicies_: Kx1: list consisting of index of assigned pilot to each UE
        H: LxKxNxrealizations, random channel realizations
        H_Mean: LxKxNxRealizations Rician LoS component (channel mean)
        p_k: 1 scalar value. Same uplink pilot transmit power for all UEs in mW

    Output:
        H_hat: LxKxNxrealizations,  channel estimates
    """

    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    realizations = params_dict["realizations"]
    p_k = np.full(K, p_k)
    H_hat = np.zeros((L, K, N, realizations), dtype=complex)
    N_tk = sqrt(0.5) * (np.random.randn(N, realizations, L, tau_p) + 1j * np.random.randn(N, realizations, L, tau_p))

    for l in range(0, L):
        for t in range(0, tau_p):
            dim_array = np.ones((1, H[l, (t == pilot_indicies_).flatten(), :, :].ndim), int).ravel()
            dim_array[0] = -1
            p_k_reshaped = p_k.reshape(dim_array)
            # received pilot signal at each AP from all UEs receiving on that pilot
            y_pilot = np.sum(
                np.sqrt(p_k_reshaped[(t == pilot_indicies_).flatten()]) * tau_p * H[l, (t == pilot_indicies_).flatten(),
                                                                                  :, :], axis=0) + sqrt(tau_p) * N_tk[:,
                                                                                                                 :, l,
                                                                                                                 t]
            y_mean = np.sum(np.sqrt(p_k_reshaped[(t == pilot_indicies_).flatten()]) * tau_p * H_Mean[l, (
                                                                                                                t == pilot_indicies_).flatten(),
                                                                                              :, :], axis=0)
            Psi = np.sum(
                p_k_reshaped[(t == pilot_indicies_).flatten()] * tau_p * R[l, (t == pilot_indicies_).flatten(), :, :],
                axis=0) + np.eye(N)

            # Go through all UEs that use pilot t
            for k in np.where(np.any(pilot_indicies_ == t, axis=1))[0]:
                # Compute the MMSE estimate
                RPsi = R[l, k, :, :] @ np.linalg.inv(Psi)
                H_hat[l, k, :, :] = H_Mean[l, k, :, :] + np.sqrt(p_k[k]) * RPsi @ (y_pilot - y_mean)

    return H_hat


def compute_SE_MMSE(params_dict, p_k, D_DCC, H_hat, H, batch=100, SE_metric="uatf"):
    """ Function to calulcate Spectral efficiency acheived by each UE, when centralized MMSE beamforming is used.
    Input:
        p_k: Kx1, uplink transmit power allocated to each UE in mW, based on any arbitrariy power control scheme
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        H_hat: LxKxNxrealizations,  channel estimates
        H: LxKxNxrealizations,  channels
    Output:
        SE_cd: Kx1 spectral efficiencies for all UEs using coherent decoding scheme in centralized uplink operation
        SE_uatf:  Kx1 spectral efficiencies for all UEs using UatF scheme in centralized uplink operation
    """
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    tau_c = params_dict["tau_c"]
    L = params_dict["L"]
    realizations = params_dict["realizations"]

    # To store values
    v_k_ = np.zeros((L, K, N, realizations), dtype=complex)  # combining vectors for all Aps and UEs
    SE_uatf = np.zeros(K)  # UatF SE for each UE
    SE_cd = np.zeros(K)  # CD SE for each UE

    # Compute the prelog factor assuming only uplink data transmission
    tau_u = tau_c - tau_p
    prelogFactor = tau_u / tau_c

    # Go through all UEs
    for k in range(0, K):
        # Compute the number of APs that serve UE k
        servingAP = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        La = len(servingAP)

        H_k = H[:, servingAP, :, :]  # RxLaxNxK  channels from only serving APs of UE k
        H_k = np.reshape(H_k, [H_k.shape[0], -1, H_k.shape[-1]])  # Rx(La.N)xK

        Hhat_k = H_hat[:, servingAP, :, :]  # RxLaxNxK  channel estimates  for only serving APs of UE k
        Hhat_k = np.reshape(Hhat_k, [Hhat_k.shape[0], -1, Hhat_k.shape[-1]])  # Rx(La.N)xK

        E_k = H_k - Hhat_k  # Rx(La.N)xK
        E_k_H = H_(E_k)  # RxKx(La.N)

        # to reduce computation time, powers of all UEs are arranged in a matrix form
        W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK
        Phi_k = np.mean(E_k @ W @ E_k_H, 0)  # La.NxLa.N

        v_k = np.zeros((realizations, La * N, 1), dtype=complex)
        for r in range(0, realizations, batch):
            # Calculating MMSE receive combining vector using eq 31. To reduce computational time of code, summations over all UEs in verse term are done in vector form. Thus equation looks different but mathematical correct and equal.
            inverse_term = np.linalg.inv(
                Hhat_k[r:r + batch] @ W[r:r + batch] @ H_(Hhat_k[r:r + batch]) + np.tile(
                    Phi_k + np.eye(La * N), (batch, 1, 1)))  # Rx(la.N)x(la.N)

            v_k[r:r + batch] = inverse_term @ Hhat_k[r:r + batch] @ np.sqrt(
                W[r:r + batch, :, k, np.newaxis])  # Rx(la.N)x1

        v_k_[servingAP, k, :, :] = np.reshape(v_k.T, [len(servingAP), N, realizations])

        # SEs acheived by MMSE beamforming is calculated using coherent decoding bound (Section 3, Eq 4) and UatF bound (Section 3, Eq 3)

        SE_uatf[k] = prelogFactor * Compute_SE_uatf(v_k, H_k, p_k, k)
        if SE_metric == "both":
            SE_cd[k] = prelogFactor * Compute_SE_cd(v_k, Hhat_k, Phi_k, p_k, k)

    if SE_metric == "both":
        SE = np.stack((SE_cd, SE_uatf))
    else:
        SE = SE_uatf
    return SE


def compute_SE_TMMSE(params_dict, p_k, D_DCC, H_hat, H, batch=100, SE_metric="uatf"):
    """ Function to calulcate Spectral efficiency acheived by each UE, when distributed Team-MMSE beamforming is used.
    Input:
        p_k: Kx1, uplink transmit power allocated to each UE in mW, based on any arbitrariy power control scheme
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        H_hat: LxKxNxrealizations,  channel estimates
        H: LxKxNxrealizations,  channels
    Output:
        SE_cd: Kx1 spectral efficiencies for all UEs using coherent decoding scheme in centralized uplink operation
        SE_uatf:  Kx1 spectral efficiencies for all UEs using UatF scheme in centralized uplink operation
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    tau_c = params_dict["tau_c"]
    realizations = params_dict["realizations"]

    # To store values
    SE_uatf = np.zeros(K)  # UatF SE for each UE
    SE_cd = np.zeros(K)  # CD SE for each UE

    # Compute the prelog factor assuming only uplink data transmission
    tau_u = tau_c - tau_p
    prelogFactor = tau_u / tau_c

    H_tilde = H - H_hat  # RxLxNxK estimation error
    # To reduce computation time, powers of all UEs are arranged in a matrix form
    W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK

    # Go through all UEs
    for k in range(0, K):
        # Compute the number of APs that serve UE k
        servingAP = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        La = len(servingAP)
        # To store Pi values required for eq 11
        big_Pi = np.zeros((La, K, K), dtype=complex)  # LaxKxK
        # To store local beamforming vector from serving APs for each UE i.e. V_l in eq 10
        T = np.zeros((realizations, La, N, K), dtype=complex)

        # For UE k, go over all serving APs
        for indx in range(0, La):
            l = servingAP[indx]
            Hhat_l = H_hat[:, l, :, :]  # RxNxK
            Hhat_l_H = H_(H_hat[:, l, :, :])  # RxKxN
            H_tilde_l_H = H_(H_tilde[:, l, :, :])  # RxKxN
            H_tilde_l = H_tilde[:, l, :, :]  # RxNxK

            C_l = np.mean(H_tilde_l @ W @ H_tilde_l_H, 0)  # NxN error correlation matrix multiplied with transmit power

            # Calculating T-MMSE receive combining vector using eq 8. To reduce computational time of code, summations over all UEs inverse term are done in vector form. Thus equation looks different but mathematical correct and equal.
            for r in range(0, realizations, batch):
                inverse_term = np.linalg.inv(
                    Hhat_l[r:r + batch] @ W[r:r + batch] @ Hhat_l_H[r:r + batch] + np.tile(C_l + np.eye(N),
                                                                                           (batch, 1, 1)))  # RxNxN
                T_l = inverse_term @ Hhat_l[r:r + batch] @ np.sqrt(W[r:r + batch])  # RxNxK
                T[r:r + batch, indx, :, :] = T_l
                big_Pi[indx, :, :] = big_Pi[indx, :, :] + np.sum(np.sqrt(W[r:r + batch]) @ Hhat_l_H[r:r + batch] @ T_l,
                                                                 0) / realizations  # KxK

        # Next few steps to solves system of linear equations given in equation 11. Resulting c_k gives a correction weight to local estimates from all serving APs.
        Pi_k = np.tile(big_Pi, (La, 1, 1, 1))  # LaxLaxKxK
        for i in range(La):
            Pi_k[i, i, :, :] = np.identity(K, dtype=complex)
        Pi_k = np.swapaxes(Pi_k, 1, 2).reshape(La * K, La * K)  # La.K x La.K
        e_k = np.zeros(K, dtype=complex)
        e_k[k] = 1
        c_k = np.linalg.solve(Pi_k, np.tile(e_k, (1, La)).T)  # Ax=b  #b La.Kx1
        c_k = c_k.reshape(La, K)  # LaxK  correction weights

        # Final beamforming vector given in eq 10
        v_k = T @ np.tile(c_k[:, :, np.newaxis], (realizations, 1, 1, 1))  # RxLaxNx1
        v_k = np.reshape(v_k, [v_k.shape[0], -1, v_k.shape[-1]])  # Rx(La.N)x1

        # Reshaping and selecting channel, estimates and errors to input them to the SE expression
        H_for_k = np.reshape(H[:, servingAP, :, :], (H.shape[0], -1, H.shape[3]))  # RxLa.NxK
        Hhat_for_k = np.reshape(H_hat[:, servingAP, :, :], (H_hat.shape[0], -1, H_hat.shape[3]))  # RxLa.NxK
        H_tilde_k = H_for_k - Hhat_for_k  # Rx(La.N)xK
        H_tilde_k_H = H_(H_tilde_k)  # RxKx(La.N)
        W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK
        C_k = H_tilde_k @ W @ H_tilde_k_H  # RxLa.NxLa

        # SEs acheived by Team-MMSE beamforming is calculated using coherent decoding bound (Section 3, Eq 4) and UatF bound (Section 3, Eq 3)
        if SE_metric == "both":
            SE_cd[k] = prelogFactor * Compute_SE_cd(v_k, Hhat_for_k, C_k, p_k, k)
        SE_uatf[k] = prelogFactor * Compute_SE_uatf(v_k, H_for_k, p_k, k)

    if SE_metric == "both":
        SE = np.stack((SE_cd, SE_uatf))
    else:
        SE = SE_uatf
    return SE


def compute_SE_LMMSE(params_dict, p_k, D_DCC, H_hat, H, batch=100, SE_metric="uatf"):
    """ Function to calulcate Spectral efficiency acheived by each UE, when distributed Local-MMSE beamforming is used.
    Input:
        p_k: Kx1, uplink transmit power allocated to each UE in mW, based on any arbitrariy power control scheme
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        H_hat: LxKxNxrealizations,  channel estimates
        H: LxKxNxrealizations,  channels
    Output:
        SE_cd: Kx1 spectral efficiencies for all UEs using coherent decoding scheme in centralized uplink operation
        SE_uatf:  Kx1 spectral efficiencies for all UEs using UatF scheme in centralized uplink operation
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    tau_c = params_dict["tau_c"]
    realizations = params_dict["realizations"]

    # To store values
    SE_uatf = np.zeros(K)  # UatF SE for each UE
    SE_cd = np.zeros(K)  # CD SE for each UE

    # Compute the prelog factor assuming only uplink data transmission
    tau_u = tau_c - tau_p
    prelogFactor = tau_u / tau_c

    # For local MMSE, code structure is followed from the description in reference [2]
    g_for_all_i, F, V = receive_vector_dist(L, K, N, realizations, D_DCC, H, H_hat, p_k, batch)
    for k in range(0, K):
        servingAPs = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        la = len(servingAPs)
        E_g_kk = np.zeros((la, 1), dtype=complex)
        E_gg_ki = np.zeros((K, la, la), dtype=complex)
        for r in range(0, realizations, batch):
            gki = np.swapaxes(g_for_all_i[r:r + batch, k, servingAPs, :], 1, 2)  # RxKxLa
            gkk = g_for_all_i[r:r + batch, k, servingAPs, k, np.newaxis]  # RxLax1
            E_g_kk = E_g_kk + np.sum(gkk, axis=0) / realizations  # lax1
            E_gg_ki = E_gg_ki + np.sum(gki[:, :, :, np.newaxis] @ np.conj(gki[:, :, np.newaxis, :]),
                                       axis=0) / realizations  # KxLaxLa
        F_k = np.diag(F[servingAPs, k])  # LaxLa real values

        # Calculating optimal LSFD weights required in eq 35 from [2]
        inverse_term_c = np.linalg.inv(np.sum(p_k.reshape([-1, 1, 1]) * E_gg_ki, axis=0) + F_k)
        c_k_opt = p_k[k] * (inverse_term_c @ E_g_kk)

        # Final beamforming vector according to eq 35 from [2]
        v_k = V[:, servingAPs, :, k] * np.tile(c_k_opt[:, :, np.newaxis], (1, realizations, N))  # RxLaxNx1
        v_k = np.moveaxis(v_k[:, :, :, np.newaxis], 0, 1)
        v_k = np.reshape(v_k, [v_k.shape[0], -1, v_k.shape[-1]])  # Rx(La.N)x1

        # Reshaping and selecting channel, estimates and errors to input them to the SE expression
        H_for_k = np.reshape(H[:, servingAPs, :, :], (H.shape[0], -1, H.shape[3]))  # RxLa.NxK
        Hhat_for_k = np.reshape(H_hat[:, servingAPs, :, :], (H_hat.shape[0], -1, H_hat.shape[3]))  # RxLa.NxK
        E_k = H_for_k - Hhat_for_k  # Rx(La.N)xK
        E_k_H = H_(E_k)  # RxKx(La.N)
        W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK
        C_k = E_k @ W @ E_k_H  # RxLa.NxLa

        # SEs acheived by Local-MMSE beamforming is calculated using coherent decoding bound (Section 3, Eq 4) and UatF bound (Section 3, Eq 3)
        if SE_metric == "both":
            SE_cd[k] = prelogFactor * Compute_SE_cd(v_k, Hhat_for_k, C_k, p_k, k)
        SE_uatf[k] = prelogFactor * Compute_SE_uatf(v_k, H_for_k, p_k, k)

    if SE_metric == "both":
        SE = np.stack((SE_cd, SE_uatf))
    else:
        SE = SE_uatf
    return SE


def receive_vector_dist(L, K, N, realizations, D_DCC, H, H_hat, p_k, batch=100):
    """ Function to compute local beamforming vector, and data structures required to calculate optimal LSFD.
         For this, code structure is followed from the description in reference [2]
    """
    F = np.zeros([L, K])  # Real values
    g_for_all_i = np.zeros([realizations, K, L, K], dtype=complex)
    V = np.zeros([realizations, L, N, K], dtype=complex)  # Real values

    H_tilde = H - H_hat  # RxLxNxK estimation error
    # to reduce computation time, powers of all UEs are arranged in a matrix form
    W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK

    # Go over all APs
    for l in range(0, L):
        servingUEs = np.asarray(np.where(D_DCC[l, :] == 1)).reshape(-1)
        Hhat_l = H_hat[:, l, :, :]  # RxNxK
        Hhat_l_H = H_(H_hat[:, l, :, :])  # RxKxN
        H_tilde_l_H = H_(H_tilde[:, l, :, :])  # RxKxN
        H_tilde_l = H_tilde[:, l, :, :]  # RxNxK
        C_l = np.mean(H_tilde_l @ W @ H_tilde_l_H, 0)  # NxN

        F_temp = np.zeros([batch, K])  # Real values

        # Calculating L-MMSE receive combining vector using eq 34 from [2]. To reduce computational time of code, summations over all UEs in verse term are done in vector form. Thus equation looks different but mathematical correct and equal.
        for r in range(0, realizations, batch):
            inverse_term = np.linalg.inv(
                Hhat_l[r:r + batch] @ W[r:r + batch] @ Hhat_l_H[r:r + batch] + np.tile(C_l + np.eye(N),
                                                                                       (batch, 1, 1)))  # RxNxN
            v_L = inverse_term @ Hhat_l[r:r + batch] @ np.sqrt(W[r:r + batch])  # RxNxK
            v_L = v_L[:, :, servingUEs]  # RxNxKa
            V[r:r + batch, l, :, servingUEs] = np.moveaxis(v_L, -1, 0)
            v_L = np.moveaxis(v_L, 2, 1)  # RxKaxN
            F_temp[:, servingUEs] = np.sum(abs(v_L) ** 2, axis=2)  # RxKax1
            F[l, :] = F[l, :] + np.sum(F_temp, axis=0) / realizations
            g_for_all_i[r:r + batch, servingUEs, l, :] = np.conj(v_L) @ H[r:r + batch, l, :, :]  # RxKaxK
    return g_for_all_i, F, V


def Compute_SE_cd(v_k, Hhat_k, Z_k, p_k, k):
    """
    Function to calculate SE using coherent decoding bound (Section 3 Eq 4) as metric
    Input:
        v_k: Rx(la.N)x1 receive combining vectors for all serving APs of kth UE for all small scale realizations
        Hhat_k: Rx(La.N)xK  channel estimates of N size between serving APs of kth UE and all other UEs for all small scale realizations
        Z_k: Rx(la.N)x(la.N) Estimation error correlation matrice of NxN size between serving APs of kth UE and all other UEs
        p_k: uplink transmit power of pilot for kth UE
        k: index of UE
    Output:
        SE: 1x1 spectral efficiency for kth UE using coherent decoding lower bound
    """
    v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
    numerator = p_k[k] * abs((v_k_H @ Hhat_k[:, :, k, np.newaxis]) ** 2).flatten()
    denominator = np.sum(p_k.T * abs((v_k_H @ Hhat_k) ** 2), axis=2).flatten() - numerator + (
            v_k_H @ Z_k @ v_k).flatten() + np.sum(abs(v_k[:, :, 0] ** 2), axis=1).flatten()
    SINR = numerator / denominator  # instantaneous SINR
    SE = np.mean(np.real(np.log2(1 + SINR)))
    return SE


def Compute_SE_uatf(v_k, H_k, p_k, k):
    """
    Function to calculate SE using UatF bound (Section 3 Eq 2) as metric
    Input:
        v_k: Rx(la.N)x1 receive combining vectors for all serving APs of kth UE for all small scale realizations
        H_k: Rx(La.N)xK  channel of N size between serving APs of kth UE and all other UEs for all small scale realizations
        p_k: uplink transmit power of pilot for kth UE
        k: index of UE
    Output:
        SE: 1x1 spectral efficency for kth UE using UatF lower bound
    """
    v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
    numerator = (p_k[k] * abs(np.mean(v_k_H @ H_k[:, :, k, np.newaxis], axis=0) ** 2)).flatten()
    denominator = np.sum(p_k.T * np.mean(abs((v_k_H @ H_k) ** 2), axis=0), axis=1).flatten() + np.mean(
        np.sum(abs(v_k[:, :, 0] ** 2), axis=1)) - numerator
    SINR = numerator / denominator
    SE = np.log2(1 + SINR)
    return SE


def calculate_local_scattering(N, varPhi, varTheta, varAntennaSpacing):
    """ Calculating normalized spatial correlation matrix between one AP and one UE
    Input:
        N: 1x1 total number of antennas at AP
        varPhi: 1x1 azimuth angle
        varTheta: 1x1 angle of elevation
        varAntennaSpacing: 1x1 antenna spacing from one antenna to the next in one AP
    Output:
        R: NxN Normalized spatial correlation matrix
    """
    varASD_phi = radians(5)  # angular standard deviation for phi
    varASD_theta = radians(5)  # angular standard deviation for theta
    spatial_corr_UE = np.zeros((N, 1), dtype=complex)
    spatial_corr_UE[0] = 1
    # Go through all the columns of the first row
    for n in range(1, N):
        # Calculating the joint pdf of Phi and Theta
        inside_integral_real = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                       exp(-varphiDelta ** 2 / (
                                                                               2 * varASD_phi ** 2))) * (
                                                                       exp(-varthetaDelta ** 2 / (
                                                                               2 * varASD_theta ** 2))) / (
                                                                           2 * pi * varASD_theta * varASD_phi)).real
        inside_integral_img = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                      exp(-varphiDelta ** 2 / (
                                                                              2 * varASD_phi ** 2))) * (
                                                                      exp(-varthetaDelta ** 2 / (
                                                                              2 * varASD_theta ** 2))) / (
                                                                          2 * pi * varASD_theta * varASD_phi)).imag

        res_real = integrate.dblquad(inside_integral_real, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        res_imag = integrate.dblquad(inside_integral_img, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        spatial_corr_UE[n] = res_real[0] + 1j * res_imag[0]

    # Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R = linalg.toeplitz(spatial_corr_UE.ravel())  # integral gives Nx1, we need to calculate NxN matrix from it
    R = R / np.trace(R)  # normalizing
    return R


def pilot_assignment(k_, main_AP, tau_p, pilot_index, channel_gain):  # pilot assignment using Algo 4.1 from [2]
    if k_ <= tau_p - 1:
        pilot_index[k_] = k_
    else:
        pilotinterference = np.zeros((tau_p, 1))
        for t in range(0, tau_p):
            mask = (pilot_index[:] == t).flatten()
            pilotinterference[t] = np.sum(
                db2pow(channel_gain[main_AP, mask]))  # rows where UE's main_AP using same pilot for other UEs
        pilot_index[k_] = pilotinterference.argmin()  # pilot on which there's min interference is assignd to UE
    return pilot_index


def createDCC(D_MasterAps, L, tau_p, Beta, pilot_indicies_):  # Cooperation cluster formation using Algo 4.1 from [2]
    D_ = copy.deepcopy(D_MasterAps)
    for l_ in range(0, L):
        for t in range(0, tau_p):
            ue_mask = (pilot_indicies_[:] == t).flatten().astype(int)
            ue_mask[ue_mask == 0] = -1
            ue_index = (db2pow(Beta[l_, :]) * ue_mask).argmax()
            D_[l_, ue_index] = 1
    return D_


def db2pow(ydb):  # converts dB to power
    return np.power(10, ydb / 10)


def pow2db(y):  # converts power to dB
    return 10 * np.log10(y)


def H_(vec):  # Hermitian of more than 2-dimensional  matrix
    dim = vec.ndim - 1
    vec_H = np.conj(np.swapaxes(vec, dim, dim - 1))
    return vec_H


def read_SEs_dist_(results_path, K, distances, simulation_num=100, DCC='DCC', metric="uatf"):
    "Not part of uplink operation. Just required for reading saved data after running simulations to plot graphs"
    if metric == "both":
        SE_dim = 2
    else:
        SE_dim = 1
    tmmse_se = np.ones((len(distances), simulation_num, SE_dim, K), float)
    lmmse_se = np.zeros((len(distances), simulation_num, SE_dim, K), float)
    mmse_se = np.zeros((len(distances), simulation_num, SE_dim, K), float)

    d = 0
    for dist in distances:
        if dist != "0.1":
            dist = int(dist)
        for i in range(0, simulation_num):
            if os.path.exists(results_path + 'SE_MMSE_' + DCC + '_' + str(i) + "_" + str(dist) + '.npy'):
                with open(results_path + 'SE_MMSE_' + DCC + '_' + str(i) + "_" + str(dist) + '.npy', 'rb') as f:
                    mmse_se[d, i] = np.load(f, allow_pickle=True)
                with open(results_path + 'SE_LMMSE_' + DCC + '_' + str(i) + "_" + str(dist) + '.npy', 'rb') as f:
                    lmmse_se[d, i] = np.load(f, allow_pickle=True)
                with open(results_path + 'SE_TMMSE_' + DCC + '_' + str(i) + "_" + str(dist) + '.npy', 'rb') as f:
                    tmmse_se[d, i] = np.load(f, allow_pickle=True)
        d += 1
    return mmse_se, lmmse_se, tmmse_se
