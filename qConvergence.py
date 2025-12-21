from types import SimpleNamespace
import numpy as np
from matplotlib import pyplot as plt

class InvestorForecast:

    def __init__(self):

        par = self.par = SimpleNamespace()

        par.pi_L_pos = 0.581  #sandsynlighed for to ens stød i regime 1 når gammelt y er positivt
        par.pi_L_neg = 0.079  #sandsynlighed for to ens stød i regime 1 når gammelt y er negativt
        par.pi_H_pos = 0.804  #sandsynlighed for to ens stød i regime 2 når gammelt y er positivt
        par.pi_H_neg = 0.478  #sandsynlighed for to ens stød i regime 2 når gammelt y er negativt
        par.lam_1 = 0.555  #sandsynlighed for regime 1 -> regime 2
        par.lam_2 = 0.594  #sandsynlighed for regime 2 -> regime 1
        par.q_ini = 0.5
    
    def forecast(self, old_y, new_y, q):
        if old_y >= 0:
            pi_L = self.par.pi_L_pos
            pi_H = self.par.pi_H_pos
        else:
            pi_L = self.par.pi_L_neg
            pi_H = self.par.pi_H_neg

        if old_y == new_y:
            new_q = (pi_L * ((1-self.par.lam_1) * q + self.par.lam_2 * (1-q))) / (pi_L * ((1-self.par.lam_1) * q + self.par.lam_2 * (1-q)) + pi_H * (self.par.lam_1*q + (1-self.par.lam_2) * (1-q)))
        else:
            new_q = ((1-pi_L) * ((1-self.par.lam_1) * q + self.par.lam_2 * (1-q))) / ((1-pi_L) * ((1-self.par.lam_1) * q + self.par.lam_2 * (1-q)) + (1-pi_H) * (self.par.lam_1 * q + (1-self.par.lam_2) * (1 - q)))
        return new_q
    
    def convergence(self, old_y, new_y, conDetails = True):
    
    # conDetails tillader test af konvergens uden graf og tekst, hvis den sættes til False
        max_iterations = 100
        tolerance = 1e-6
        q_values = [self.par.q_ini]
        q_exante_vals = []
        if conDetails:
            print("Iteration\tq\t\tnew_q")
        for i in range(max_iterations):
            old_q = q_values[-1]
            q_exante = old_q * (1-self.par.lam_1) + (1-old_q) * self.par.lam_2
            q_exante_vals.append(q_exante)
            new_q = self.forecast(old_y, new_y, old_q)
            q_values.append(new_q)
            if conDetails:
                print(f"{i+1}\t\t{old_q:.6f}\t{new_q:.6f}\t{q_exante:.6f}")
            if abs(new_q - old_q) < tolerance:
                break
        
        if conDetails:
            plt.plot(range(len(q_values)), q_values, marker='o', linestyle='-')
            plt.plot(range(1, len(q_values)), q_exante_vals, marker='o', linestyle='--')
            plt.xlabel('Iteration')
            plt.ylabel('q Value')
            plt.title('Convergence of q')
            plt.grid(True)
            plt.show()

        return q_values
    
    

    def simulate(self, n, simDetails = True):
        # simDetails tillader at køre simulation uden grafer og tekst, hvis sættes til False
        old_y = 1
        q_values = []
        q_exante = []
        q = self.par.q_ini
        for i in range(n):
            new_y = np.random.choice([-1, 1])
            q_values.append(self.forecast(old_y,new_y,q))
            if simDetails:
                print(f"Period {i+1:2}: old_y = {old_y:2}, new_y = {new_y:2}, q = {q_values[i]:.2f}")
            old_y = new_y
            q = q_values[i]
            q_exante.append(q_values[i] * (1- self.par.lam_1) + (1-q_values[i]) * self.par.lam_2)
            
            
        if simDetails:
            plt.plot(range(1, n+1), q_values, marker='o', linestyle='--')
            plt.plot(range(2, n+2), q_exante, marker='o', linestyle='--')
            plt.xlabel('Iteration')
            plt.ylabel('q Value')
            plt.title('q Value for Each Iteration')
            plt.grid(True)
            plt.show()

        return q_values