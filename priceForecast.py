from types import SimpleNamespace
import numpy as np
from matplotlib import pyplot as plt

class InvestorForecast:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # Eksogene parametre
        par.r = 0.0035 # rente
        par.V_gamma = 0.00246 # varians af log-afkast
        par.a = 1 # risikoaversion

        # Initiale parametre
        par.q_ini = 0.5
        par.nF_ini = 0.5
        par.nC_ini = 1 - par.nF_ini

        # Parametre fra MS-AR(1) model
        par.pi_L_pos = 0.581  #sandsynlighed for to ens stød i regime 1 når gammelt y er positivt
        par.pi_L_neg = 0.079  #sandsynlighed for to ens stød i regime 1 når gammelt y er negativt
        par.pi_H_pos = 0.804  #sandsynlighed for to ens stød i regime 2 når gammelt y er positivt
        par.pi_H_neg = 0.478  #sandsynlighed for to ens stød i regime 2 når gammelt y er negativt
        par.lam_1 = 0.555  #sandsynlighed for regime 1 -> regime 2
        par.lam_2 = 0.594  #sandsynlighed for regime 2 -> regime 1

        # Chartist AR-parametre
        par.mu_neg = 0.0287
        par.phi_neg = -0.582

        par.mu_pos = -0.0168
        par.phi_pos = 0.587

        # Fundamentalist forventning
        par.gamma_fund = 0.008166
    


        # Forventede prisstigninger i log-afkast
    def expected_gamma_neg(self, gamma_old):
        par = self.par
        return par.mu_neg + par.phi_neg * gamma_old
    
    def expected_gamma_pos(self, gamma_old):
        par = self.par            
        return par.mu_pos + par.phi_pos * gamma_old
        
    def expected_gamma_Fund(self):
        return self.par.gamma_fund
        
    # Efterspørgselsfunktioner
    def demand(self, E_gamma, P_old):
        par = self.par

        return (E_gamma - par.r) / (par.a * P_old * par.V_gamma)
        
    # Clearingspris
    def clearing_price(self, nNeg, nPos, nF, E_Ppos, E_Pneg, E_PF):
        par = self.par
        return (nNeg * E_Pneg + nPos * E_Ppos + nF * E_PF) / (1 + par.r)
        
    # Profitfunktion
    def profit(self, new_P, old_P, z):
        return z * (new_P - old_P * (1.0 + self.par.r))
        
    # Diskret valgmodel - returnerer andelen af fundamentalister til næste periode
    def discrete_choice(self, pi_C, pi_F):
        exp_C = np.exp(pi_C)
        exp_F = np.exp(pi_F)
        denom = exp_C + exp_F
        return exp_F / denom
        
    # Bayesiansk opdateringsmekanisme
    def update_q(self, old_gamma, new_gamma, old_q):
        par = self.par
        # pi_L og pi_H vælges pba old_gamma
        if old_gamma >= 0:
            pi_L = par.pi_L_pos
            pi_H = par.pi_H_pos
        else:
            pi_L = par.pi_L_neg
            pi_H = par.pi_H_neg
            
        # Ex ante sandsynlighed for regime 1 og 2
        prior_reg1 = old_q * (1-par.lam_1) + (1-old_q) * par.lam_2
        prior_reg2 = old_q * par.lam_1 + (1-old_q) * (1-par.lam_2)

        # Undersøger fortegn for stød
        same_sign = (old_gamma >= 0 and new_gamma >= 0) or (old_gamma < 0 and new_gamma < 0)

        # Ex post opdatering
        if same_sign:
            num = pi_L * prior_reg1
            den = pi_L * prior_reg1 + pi_H * prior_reg2
        else:
            num = (1-pi_L) * prior_reg1
            den = (1-pi_L) * prior_reg1 + (1-pi_H) * prior_reg2
            
        return num / den
        
    # Ex ante forventninger til næste periode
    def ex_ante_prob_1(self, ex_post_q):
        par = self.par
        return ex_post_q * (1-par.lam_1) + (1-ex_post_q) * par.lam_2

    # Simulation
    def run(self, P_init, gamma_init, T, nF_ini = None, q_start = None):
        par = self.par

        # Initialisering
        nF = self.nF_ini if nF_ini is None else nF_ini
        nC = 1 - nF

        q_prev = self.q_ini if q_start is None else q_start

        # Ex ante split af chartister
        prob1 = self.ex_ante_prob_1(q_prev)
        nNeg = nC * prob1
        nPos = nC * (1-prob1)

        # Output arrays
        P_path = np.zeros(T)
        gamma_path = np.zeros(T)
        q_path = np.zeros(T)
        nF_path = np.zeros(T)
        nC_path = np.zeros(T)
        nNeg_path = np.zeros(T)
        nPos_path = np.zeros(T)

        P = P_init
        gamma = gamma_init

        for t in range(T):
            P_old = P
            gamma_old = gamma

            # Forventninger
            E_gamma_neg = self.expected_gamma_neg(gamma_old)
            E_gamma_pos = self.expected_gamma_pos(gamma_old)
            E_gamma_fund = self.expected_gamma_Fund()

            E_Pneg = P_old * (1 + E_gamma_neg)
            E_Ppos = P_old * (1 + E_gamma_pos)
            E_PF = P_old * (1 + E_gamma_fund)

            # Efterspørgsler
            zNeg = self.demand(E_gamma_neg, P_old)
            zPos = self.demand(E_gamma_pos, P_old)
            zF = self.demand(E_gamma_fund, P_old)

            # Clearing pris
            P_new = self.clearing_price(nNeg, nPos, nF, E_Pneg, E_Ppos, E_PF)
            gamma_new = np.log(P_new) - np.log(P_old)

            # Profitter
            pi_neg = self.profit(P_new, P_old, zNeg)
            pi_pos = self.profit(P_new, P_old, zPos)
            pi_F = self.profit(P_old, P_new, zF)

            # Evolutionær opdatering
            nF_next = self.discrete_choice((pi_neg+pi_pos), pi_F)
            nC_next = 1.0 - nF_next

            # Bayesiansk opdatering af ex post q
            q_expost_t = self.update_q(gamma_old, gamma_new, q_prev)

            # Opdatering af ex ante q
            P1_exante = self.ex_ante_prob_1(q_expost_t)
            nNeg_next = nC_next * P1_exante
            nPos_next = nC_next * (1-P1_exante)

            # Gemt output
            P_path[t] = P_new
            gamma_path[t] = gamma_new
            q_path[t] = q_expost_t
            nF_path[t] = nF
            nC_path[t] = nC
            nNeg_path[t] = nNeg
            nPos_path[t] = nPos

            # Opdatering til næste iteration
            P = P_new
            gamma = gamma_new
            q_prev = q_expost_t
            nF = nF_next
            nC = nC_next
            nNeg = nNeg_next
            nPos = nPos_next
        
        return {
        "P": P_path,
        "gamma": gamma_path,
        "q": q_path,
        "nF": nF_path,
        "nC": nC_path,
        "nNeg": nNeg_path,
        "nPos": nPos_path
        }