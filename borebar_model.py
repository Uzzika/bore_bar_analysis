import numpy as np
from scipy.optimize import root_scalar

class BoreBarModel:
    @staticmethod
    def calculate_torsional(params):
        """Расчет крутильных колебаний"""
        rho = params['rho']
        G = params['G']
        Jp = params['Jp']
        Jr = params['Jr']
        delta1 = params['delta1'] * params['multiplier']
        length = params['length']

        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = length * np.sqrt(rho / G)

        omega = np.linspace(100, 15000, 5000)
        p = 1j * omega

        with np.errstate(all='ignore'):
            expr = np.sqrt(1 + delta1 * p)
            coth_arg = lambda2 * p / expr
            coth_arg = np.where(np.abs(coth_arg) > 100, 100 * np.sign(coth_arg), coth_arg)
            cth = (np.exp(2 * coth_arg) + 1) / (np.exp(2 * coth_arg) - 1)
            cth = np.nan_to_num(cth, nan=1.0, posinf=1.0, neginf=-1.0)
            sigma = -p - lambda1 * expr * cth

        valid = (np.abs(sigma.real) < 1e6) & (np.abs(sigma.imag) < 1e6)
        # print(f"[DEBUG] λ₁ = {lambda1:.3e}, λ₂ = {lambda2:.3e}, δ₁ = {delta1:.3e}")
        # print(f"[DEBUG] Valid points: {np.sum(valid)} / {len(sigma)}")
        # print(f"[DEBUG] Re(σ) min/max: {np.min(sigma.real):.3e} / {np.max(sigma.real):.3e}")
        # print(f"[DEBUG] Im(σ) min/max: {np.min(sigma.imag):.3e} / {np.max(sigma.imag):.3e}")

        return {
            'omega': omega[valid],
            'sigma_real': sigma.real[valid],
            'sigma_imag': sigma.imag[valid],
            'lambda1': lambda1,
            'lambda2': lambda2,
            'delta1': delta1
        }

    @staticmethod
    def calculate_longitudinal(params):
        """Расчет продольных колебаний"""
        E = params['E']
        S = params['S']
        rho = params['rho']
        L = params['length']
        mu = params['mu']
        tau = params['tau']

        a2 = E / rho
        omega_main = np.pi*a2/L
        omega = np.linspace(0.01, 2*np.pi*100, 5000)

        with np.errstate(all='ignore'):
            x = omega * L / a2
            mask = (np.abs(np.sin(x)) > 1e-6)
            cot = np.zeros_like(x)
            cot[mask] = 1/np.tan(x[mask])
            
            denom = 1 - mu * np.cos(omega * tau)
            denom_mask = np.abs(denom) > 1e-6
            
            valid = mask & denom_mask
            
            K1 = np.full_like(omega, np.nan)
            delta = np.full_like(omega, np.nan)
            
            K1[valid] = (E*S/a2) * omega[valid] * cot[valid] / denom[valid]
            delta[valid] = -(E*S*mu/a2) * cot[valid] * np.sin(omega[valid]*tau) / denom[valid]
            
            valid = valid & (K1 > 0) & (K1 < 1e10) & (np.abs(delta) < 1e6)
            
        return {
            'omega': omega[valid],
            'K1': K1[valid],
            'delta': delta[valid],
            'a': a2,
            'omega_main': omega_main,
            'K1_0': (E*S)/(L*(1 - mu)),
            'delta_0': - (E*S*mu*tau)/(L*(1 - mu))
        }

    @staticmethod
    def calculate_comparative(params):
        """Сравнительный анализ"""
        lengths = np.linspace(2, 6, 20)
        torsional_freq = np.sqrt(params['G']/params['rho']) * np.pi / lengths
        longitudinal_freq = np.sqrt(params['E']/params['rho']) * np.pi / lengths
        stability_ratio = 1 / lengths**2
        
        return {
            'lengths': lengths,
            'torsional_freq': torsional_freq,
            'longitudinal_freq': longitudinal_freq,
            'stability_ratio': stability_ratio
        }

    @staticmethod
    def find_intersection(params):
        """Поиск точки пересечения с осью Im(σ) = 0 для крутильных колебаний"""
        rho = params['rho']
        G = params['G']
        Jp = params['Jp']
        Jr = params['Jr']
        delta1 = params['delta1'] * params['multiplier']
        length = params['length']

        lambda1 = np.sqrt(rho * G) * Jp / Jr
        lambda2 = length * np.sqrt(rho / G)

        def im_sigma(omega_val):
            p_val = 1j * omega_val
            with np.errstate(all='ignore'):
                sqrt_expr = np.sqrt(1 + delta1 * p_val)
                coth_arg = lambda2 * p_val / sqrt_expr
                coth_arg = np.where(np.abs(coth_arg) > 100, 100 * np.sign(coth_arg), coth_arg)
                cth = (np.exp(2 * coth_arg) + 1) / (np.exp(2 * coth_arg) - 1)
                cth = np.nan_to_num(cth, nan=1.0, posinf=1.0, neginf=-1.0)
                return (-p_val - lambda1 * sqrt_expr * cth).imag

        try:
            brackets = [(500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
            omega_cross = None
            for bracket in brackets:
                try:
                    sol = root_scalar(im_sigma, bracket=bracket, method='brentq')
                    if sol.converged:
                        omega_cross = sol.root
                        break
                except:
                    continue

            if omega_cross is None:
                return None

            p_cross = 1j * omega_cross
            sqrt_expr = np.sqrt(1 + delta1 * p_cross)
            coth_arg = lambda2 * p_cross / sqrt_expr
            coth_arg = np.where(np.abs(coth_arg) > 100, 100 * np.sign(coth_arg), coth_arg)
            cth = (np.exp(2 * coth_arg) + 1) / (np.exp(2 * coth_arg) - 1)
            cth = np.nan_to_num(cth, nan=1.0, posinf=1.0, neginf=-1.0)
            sigma_val = -p_cross - lambda1 * sqrt_expr * cth

            return {
                'omega': omega_cross,
                're_sigma': sigma_val.real,
                'frequency': omega_cross / (2 * np.pi)
            }
        except Exception:
            return None
