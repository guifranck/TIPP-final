"""
TIPP Project: Computation of level densities and photon strength functions in atomic nuclei 
Authors: Martí Serret and Guillaume Franck
Date: February 2026

Main functionalities:
1. Level density rho(E)
2. Spectrum plot
3. Rho vs Bethe formula
4. Averaged B(XL) transitions
5. Gamma Strength Function F (MeV^-3)
6. TALYS format export (mb/MeV) - both parities merged
7. Brink-Axel hypothesis check
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion() 


class TIPP:

    def __init__(self):

        # CONSTANTES
        self.C_M = 1.1540257488758977e-08   # M1
        self.C_E = 1.046735373130066e-06    # E1 

        # DICTIONNARY OF ELEMENTS
        # Each entry with parities has "+" and/or "-" sub-entries:
        #   (spec_file, transition_file, prefactor_C, transition_type)
        self.elements = {
            "1": {
                "name": "Te130",
                "Z": 52,
                "A": 130,
                "parities": {
                    "+": ("te130pos_specj.txt", "te130pos_m1.txt", self.C_M, "M1"),
                    "-": ("te130neg_specj.txt", "te130neg_m1.txt", self.C_M, "M1"),
                }
            },
            "2": {
                "name": "Xe134",
                "Z": 54,
                "A": 134,
                "parities": {
                    "+": ("xe134pos_specj.txt", "xe134pos_m1.txt", self.C_M, "M1"),
                    "-": ("xe134neg_specj.txt", "xe134neg_m1.txt", self.C_M, "M1"),
                }
            },
            "3": {
                "name": "Sc44",
                "Z": 21,
                "A": 44,
                "parities": {
                    "all": ("sc44all_specj.txt", "sc44e1.txt", self.C_E, "E1"),
                }
            },
            "4": {
                "name": "Ne26",
                "Z": 10,
                "A": 26,
                "parities": {
                    "+": ("ne26pos_SPEC", "ne26pos_M1", self.C_M, "M1"),
                }
            },
        }

        # DATA STORAGE 
        self.current_element = None
        self.current_parity = None
        self.C = None
        self.transition = None
        self.Z = None
        self.A = None

        # SPECJ
        self.Jspec = None
        self.Eexc = None
        self.Jspec_val = None

        # TRANS
        self.J_parent = None
        self.B_parent = None
        self.E_parent_exc = None
        self.Eg_abs = None

        self.dE = None
        self.data_loaded = False

    # ------------------------------------------------------------------
    # MENU
    # ------------------------------------------------------------------

    def menu_principal(self):
        print("=" * 40)
        print("       MAIN MENU")
        print("=" * 40)
        print("1. Level density  rho")
        print("2. Spectrum plot")
        print("3. Rho vs Bethe formula")
        print("4. Averaged B(XL) transitions")
        print("5. Gamma Strength Function  F (MeV^-3)")
        print("6. TALYS export  F (mb/MeV)  [both parities]")
        print("7. Brink-Axel hypothesis check")
        print()
        print("9. Load / change data")
        print("0. Exit")
        print("=" * 40)

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------

    def data_load(self):
        """Load spectrum and transition files for the chosen nucleus/parity."""

        print("\nAvailable nuclei:")
        for key, nuc in self.elements.items():
            parities = list(nuc["parities"].keys())
            print(f"  {key}. {nuc['name']}  (parities: {parities})")

        choice = input("\nChoose nucleus (number): ").strip()
        if choice not in self.elements:
            print("Invalid choice.")
            return False

        element_config = self.elements[choice]
        self.current_element = element_config["name"]
        self.Z = element_config["Z"]
        self.A = element_config["A"]

        available_parities = list(element_config["parities"].keys())
        if len(available_parities) == 1:
            parity = available_parities[0]
            print(f"Single parity available: {parity} — selected automatically.")
        else:
            parity = input(f"Choose parity {available_parities}: ").strip()
            if parity not in element_config["parities"]:
                print("Invalid parity.")
                return False

        self.current_parity = parity
        fichier_spec, fichier_trans, self.C, self.transition = \
            element_config["parities"][parity]

        self.dE = float(input("Choose bin size dE (MeV): ").strip())

        # LOAD SPECTRUM
        data_spec = np.genfromtxt(fichier_spec)
        self.Jspec = data_spec[:, 0]
        Espec = data_spec[:, 1]
        Efond = np.min(Espec)
        self.Eexc = Espec - Efond
        self.Jspec_val = np.unique(self.Jspec)

        # LOAD TRANSITIONS
        data_trans = np.genfromtxt(fichier_trans)
        Ji = data_trans[:, 0].astype(int)
        Jf = data_trans[:, 1].astype(int)
        Ei = data_trans[:, 2]
        Eg = data_trans[:, 3]
        Bif = data_trans[:, 4]
        Bfi = data_trans[:, 5]
        Ef = Ei - Eg

        self.J_parent = np.where(Eg > 0, Ji, Jf)
        self.B_parent = np.where(Eg > 0, Bif, Bfi)
        E_parent = np.where(Eg > 0, Ei, Ef)
        self.E_parent_exc = E_parent - Efond
        self.Eg_abs = np.abs(Eg)

        self.data_loaded = True
        print(f"\nData loaded: {self.current_element}  parity={parity}")
        print(f"  Spectrum levels  : {len(self.Eexc)}")
        print(f"  Transitions      : {len(self.Eg_abs)}")
        print(f"  Eexc range       : [{self.Eexc.min():.2f}, {self.Eexc.max():.2f}] MeV")
        print(f"  Spin values (J)  : {self.Jspec_val / 2}")
        return True

    # ------------------------------------------------------------------
    # HELPER: ask energy and spin filters
    # ------------------------------------------------------------------

    def _ask_filters(self, label="Eexp"):
        """Ask the user for Eexp range and spin range (Jmin to Jmax)."""
        Emin_avail = self.Eexc.min()
        Emax_avail = self.Eexc.max()
        print(f"\n  Eexc range available: [{Emin_avail:.2f}, {Emax_avail:.2f}] MeV")
        emin_input = input(f"  Enter {label}_min (MeV) or Enter for {Emin_avail:.2f}: ").strip()
        emax_input = input(f"  Enter {label}_max (MeV) or Enter for {Emax_avail:.2f}: ").strip()
        Emin = float(emin_input) if emin_input != "" else Emin_avail
        Emax = float(emax_input) if emax_input != "" else Emax_avail

        print(f"  Available spins (J): {self.Jspec_val / 2}")
        jmin_input = input("  Enter J_min (or press Enter for all spins): ").strip()
        if jmin_input == "":
            selected_spins = self.Jspec_val
        else:
            Jmin = float(jmin_input) * 2
            Jmax = float(input("  Enter J_max: ").strip()) * 2
            selected_spins = [J for J in self.Jspec_val if Jmin <= J <= Jmax]
            print(f"  Selected spins (2J): {selected_spins}")

        return Emin, Emax, selected_spins

    # ------------------------------------------------------------------
    # HELPER: compute rho
    # ------------------------------------------------------------------

    def _compute_rho(self, Emin, Emax, selected_spins):
        """
        Returns:
            E_centers  (nE,)
            count_rho  (nE,)
            rho_total  (nE,)   = count_rho / dE
        """
        Ebins = np.arange(Emin, Emax + self.dE, self.dE)
        nE = len(Ebins) - 1
        count_rho = np.zeros(nE)

        spin_mask = np.isin(self.Jspec, selected_spins)

        for i in range(nE):
            cond = (self.Eexc >= Ebins[i]) & (self.Eexc < Ebins[i + 1]) & spin_mask
            count_rho[i] = np.sum(cond)

        E_centers = Ebins[:-1] + self.dE / 2
        rho_total = count_rho / self.dE
        return E_centers, count_rho, rho_total

    # ------------------------------------------------------------------
    # HELPER: compute F
    # ------------------------------------------------------------------

    def _compute_F(self, Emin_exp, Emax_exp, selected_spins, dE_use=None):
        """
        Compute the gamma strength function F(Eg).
        Follows exactly the logic of the original script with range filtering.
        """
        dE = dE_use if dE_use is not None else self.dE

        # BUILD RHO ON THE FILTERED E/J RANGE
        Jmin = min(selected_spins)
        Jbin = np.arange(Jmin, max(selected_spins) + 2, 2)
        nJspec = len(Jbin)

        Eexc_bins = np.arange(Emin_exp, Emax_exp + dE, dE)
        nEbin = len(Eexc_bins) - 1

        rho = np.zeros((nEbin, nJspec))
        for i in range(nEbin):
            for j, Jval in enumerate(Jbin):
                cond = (self.Eexc >= Eexc_bins[i]) & (self.Eexc < Eexc_bins[i+1]) & (self.Jspec == Jval)
                rho[i, j] = np.sum(cond) / dE

        # FILTER TRANSITIONS
        trans_mask = (
            (self.E_parent_exc >= Emin_exp) &
            (self.E_parent_exc < Emax_exp) &
            np.isin(self.J_parent, selected_spins)
        )

        if np.sum(trans_mask) == 0:
            print("  Warning: no transitions found for these filters.")
            return None, None, None

        Eg_abs       = self.Eg_abs[trans_mask]
        B_parent     = self.B_parent[trans_mask]
        E_parent_exc = self.E_parent_exc[trans_mask]
        J_parent     = self.J_parent[trans_mask]

        # Eg BINS
        Eg_bins = np.arange(np.min(Eg_abs), np.max(Eg_abs) + dE, dE)
        nEg = len(Eg_bins) - 1

        # COMPUTE F FOR EACH TRANSITION
        F_parent = np.zeros(len(Eg_abs))
        for i in range(len(E_parent_exc)):
            E_p   = E_parent_exc[i]
            J_p   = J_parent[i]
            B_val = B_parent[i]

            bin_E = int((E_p - Emin_exp) / dE)
            if bin_E >= nEbin:
                bin_E = nEbin - 1
            if bin_E < 0:
                bin_E = 0

            bin_J = int((J_p - Jmin) / 2)
            if bin_J >= nJspec or bin_J < 0:
                continue

            F_parent[i] = self.C * B_val * rho[bin_E, bin_J]

        # AVERAGE F BY Eg BIN
        F_sum   = np.zeros(nEg)
        count_F = np.zeros(nEg)
        for j in range(nEg):
            for i in range(len(F_parent)):
                if Eg_abs[i] >= Eg_bins[j] and Eg_abs[i] < Eg_bins[j+1]:
                    count_F[j] += 1
                    F_sum[j]   += F_parent[i]

        F_avg = np.where(count_F > 0, F_sum / count_F, 0.0)
        Eg_centers = Eg_bins[:-1] + dE / 2
        return Eg_centers, count_F, F_avg

    # ------------------------------------------------------------------
    # OPTION 1 — Level density rho(E)
    # ------------------------------------------------------------------

    def option_1_level_density(self):
        print("\n" + "-" * 50)
        print("OPTION 1: LEVEL DENSITY  rho(E)")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        Emin, Emax, selected_spins = self._ask_filters()
        E_centers, count_rho, rho_total = self._compute_rho(Emin, Emax, selected_spins)

        # SAVE
        filename = (f"rho_{self.current_element}_{self.current_parity}"
                    f"_dE{self.dE:.3f}.txt")
        header = "Eexc(MeV)   count_rho   rho(MeV^-1)"
        np.savetxt(filename,
                   np.column_stack((E_centers, count_rho, rho_total)),
                   fmt=["%.4f", "%.0f", "%.6e"],
                   delimiter="\t",
                   header=header,
                   comments="# ")
        print(f"\nData saved: {filename}")

        # PLOT
        plt.figure(figsize=(9, 5))
        plt.plot(E_centers, rho_total, drawstyle="steps-mid", linewidth=2)
        plt.yscale('log')
        plt.xlabel("Excitation energy  $E_{exc}$ (MeV)", fontsize=12)
        plt.ylabel(r"$\rho(E)$  (MeV$^{-1}$)", fontsize=12)
        plt.title(f"Level density  —  {self.current_element}  ({self.current_parity})", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # OPTION 2 — Spectrum plot
    # ------------------------------------------------------------------

    def option_2_spectrum(self):
        print("\n" + "-" * 50)
        print("OPTION 2: SPECTRUM PLOT")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        plt.figure(figsize=(5, 10))
        plt.hlines(self.Eexc, 0.2, 0.8, linewidth=0.6, color="black")
        plt.xlim(0, 1)
        plt.ylim(self.Eexc.min() - 0.2, self.Eexc.max() + 0.2)
        plt.ylabel("Excitation energy  $E_{exc}$ (MeV)", fontsize=12)
        plt.title(f"{self.current_element}  ({self.current_parity})  —  all spins", fontsize=13)
        plt.xticks([])
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Total levels: {len(self.Eexc)}")

    # ------------------------------------------------------------------
    # OPTION 3 — rho vs Bethe formula
    # ------------------------------------------------------------------

    def option_3_bethe(self):
        print("\n" + "-" * 50)
        print("OPTION 3: RHO vs BETHE FORMULA  (a = A/10)")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        Emin, Emax, selected_spins = self._ask_filters()
        E_centers, count_rho, rho_total = self._compute_rho(Emin, Emax, selected_spins)

        # BETHE: rho(E) = sqrt(pi)*exp(2*sqrt(a*E)) / (12 * a^(1/4) * E^(5/4))
        a = self.A / 10.0
        E_bethe = np.linspace(0.5, Emax, 300)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_bethe = (np.sqrt(np.pi) * np.exp(2 * np.sqrt(a * E_bethe))) / \
                        (12 * a ** 0.25 * E_bethe ** 1.25)

        # PLOT
        plt.figure(figsize=(9, 5))
        plt.plot(E_centers, rho_total, drawstyle="steps-mid", linewidth=2, label="Shell-model")
        plt.plot(E_bethe, rho_bethe, "-", linewidth=2, color="tomato",
                 label=f"Bethe  (a = A/10 = {a:.1f} MeV$^{{-1}}$)")
        plt.yscale('log')
        plt.xlabel("Excitation energy  $E_{exc}$ (MeV)", fontsize=12)
        plt.ylabel(r"$\rho(E)$  (MeV$^{-1}$)", fontsize=12)
        plt.title(f"Level density vs Bethe  —  {self.current_element}  ({self.current_parity})",
                  fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # OPTION 4 — Averaged B(XL) transitions
    # ------------------------------------------------------------------

    def option_4_averages_B(self):
        print("\n" + "-" * 50)
        print(f"OPTION 4: AVERAGED B({self.transition}) TRANSITIONS")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        Emin_avail = self.E_parent_exc.min()
        Emax_avail = self.E_parent_exc.max()
        print(f"\n  Eexp range available: [{Emin_avail:.2f}, {Emax_avail:.2f}] MeV")
        emin_input = input(f"  Enter Eexp_min (MeV) or Enter for {Emin_avail:.2f}: ").strip()
        emax_input = input(f"  Enter Eexp_max (MeV) or Enter for {Emax_avail:.2f}: ").strip()
        Emin = float(emin_input) if emin_input != "" else Emin_avail
        Emax = float(emax_input) if emax_input != "" else Emax_avail

        J_unique = np.unique(self.J_parent)
        print(f"  Available spins (J): {J_unique / 2}")
        jmin_input = input("  Enter J_min (or press Enter for all spins): ").strip()
        if jmin_input == "":
            selected_spins = J_unique
        else:
            Jmin = float(jmin_input) * 2
            Jmax = float(input("  Enter J_max: ").strip()) * 2
            selected_spins = [J for J in J_unique if Jmin <= J <= Jmax]
            print(f"  Selected spins (2J): {selected_spins}")

        # FILTER
        mask = (
            (self.E_parent_exc >= Emin) &
            (self.E_parent_exc < Emax) &
            np.isin(self.J_parent, selected_spins)
        )

        Eg_sel = self.Eg_abs[mask]
        B_sel = self.B_parent[mask]

        if len(Eg_sel) == 0:
            print("No transitions found for these filters.")
            return

        # BIN BY Eg
        Eg_bins = np.arange(Eg_sel.min(), Eg_sel.max() + self.dE, self.dE)
        nEg = len(Eg_bins) - 1
        B_avg = np.zeros(nEg)
        count_B = np.zeros(nEg)

        for i in range(nEg):
            cond = (Eg_sel >= Eg_bins[i]) & (Eg_sel < Eg_bins[i + 1])
            vals = B_sel[cond]
            count_B[i] = len(vals)
            if len(vals) > 0:
                B_avg[i] = np.mean(vals)

        Eg_centers = Eg_bins[:-1] + self.dE / 2

        # SAVE
        filename = (f"B_{self.transition}_{self.current_element}_{self.current_parity}"
                    f"_dE{self.dE:.3f}.txt")
        header = "Eg(MeV)   count_B   B_avg"
        np.savetxt(filename,
                   np.column_stack((Eg_centers, count_B, B_avg)),
                   fmt=["%.4f", "%.0f", "%.6e"],
                   delimiter="\t",
                   header=header,
                   comments="# ")
        print(f"\nData saved: {filename}")

        # PLOT
        valid = count_B > 0
        plt.figure(figsize=(9, 5))
        plt.plot(Eg_centers[valid], B_avg[valid], drawstyle="steps-mid", linewidth=2)
        plt.xlabel(r"$E_\gamma$ (MeV)", fontsize=12)
        plt.ylabel(f"<B({self.transition})>", fontsize=12)
        plt.title(f"Averaged B({self.transition})  —  {self.current_element}  ({self.current_parity})",
                  fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # OPTION 5 — Gamma Strength Function F (MeV^-3)
    # ------------------------------------------------------------------

    def option_5_strength_function(self):
        print("\n" + "-" * 50)
        print("OPTION 5: GAMMA STRENGTH FUNCTION  F (MeV^-3)")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        Emin_avail = self.E_parent_exc.min()
        Emax_avail = self.E_parent_exc.max()
        print(f"\n  Eexp range available: [{Emin_avail:.2f}, {Emax_avail:.2f}] MeV")
        emin_input = input(f"  Enter Eexp_min (MeV) or Enter for {Emin_avail:.2f}: ").strip()
        emax_input = input(f"  Enter Eexp_max (MeV) or Enter for {Emax_avail:.2f}: ").strip()
        Emin = float(emin_input) if emin_input != "" else Emin_avail
        Emax = float(emax_input) if emax_input != "" else Emax_avail

        J_unique = np.unique(self.J_parent)
        print(f"  Available spins (J): {J_unique / 2}")
        jmin_input = input("  Enter J_min (or press Enter for all spins): ").strip()
        if jmin_input == "":
            selected_spins = J_unique
        else:
            Jmin = float(jmin_input) * 2
            Jmax = float(input("  Enter J_max: ").strip()) * 2
            selected_spins = [J for J in J_unique if Jmin <= J <= Jmax]
            print(f"  Selected spins (2J): {selected_spins}")

        Eg_centers, count_F, F_avg = self._compute_F(Emin, Emax, selected_spins)
        if Eg_centers is None:
            return

        # SAVE
        filename = (f"F_{self.transition}_{self.current_element}_{self.current_parity}"
                    f"_dE{self.dE:.3f}.txt")
        header = "Eg(MeV)   count_F   F(MeV^-3)"
        np.savetxt(filename,
                   np.column_stack((Eg_centers, count_F, F_avg)),
                   fmt=["%.4f", "%.0f", "%.6e"],
                   delimiter="\t",
                   header=header,
                   comments="# ")
        print(f"\nData saved: {filename}")

        # PLOT
        valid = count_F > 0
        plt.figure(figsize=(9, 5))
        plt.plot(Eg_centers[valid], F_avg[valid], drawstyle="steps-mid", linewidth=2)
        plt.yscale('log')
        plt.xlabel(r"$E_\gamma$ (MeV)", fontsize=12)
        plt.ylabel(f"$f_{{{self.transition}}}$  (MeV$^{{-3}}$)", fontsize=12)
        plt.title(f"Strength Function  F({self.transition})  —  "
                  f"{self.current_element}  ({self.current_parity})", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # OPTION 6 — TALYS export (both parities merged)
    # ------------------------------------------------------------------

    def option_6_talys_format(self):
        """
        Merge both parities (+/-) for the chosen nucleus and export F in
        TALYS format (mb/MeV) with 0.1 MeV bins.
        """
        print("\n" + "-" * 50)
        print("OPTION 6: TALYS EXPORT  F (mb/MeV)  — both parities")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        # FIND ELEMENT KEY
        elem_key = None
        for k, v in self.elements.items():
            if v["name"] == self.current_element:
                elem_key = k
                break

        avail = list(self.elements[elem_key]["parities"].keys())
        if len(avail) < 2:
            print(f"Only one parity available ({avail}). TALYS export will use it alone.")
        parities_to_use = avail

        C_TALYS  = 8.674      # constant [mb/MeV]
        dE_talys = 0.1

        # LOAD AND CONCATENATE BOTH PARITIES
        all_Eg    = []
        all_B     = []
        all_Ep    = []
        all_Eexc  = []

        for par in parities_to_use:
            f_spec, f_trans, C_par, trans_par = self.elements[elem_key]["parities"][par]
            try:
                ds = np.genfromtxt(f_spec)
                dt = np.genfromtxt(f_trans)
            except Exception as e:
                print(f"  Could not load files for parity {par}: {e}")
                continue

            Jsp  = ds[:, 0]
            Esp  = ds[:, 1]
            Efond = np.min(Esp)
            Eexc_p = Esp - Efond
            all_Eexc.append(Eexc_p)

            Ji  = dt[:, 0].astype(int)
            Jf  = dt[:, 1].astype(int)
            Ei  = dt[:, 2]
            Eg  = dt[:, 3]
            Bif = dt[:, 4]
            Bfi = dt[:, 5]
            Ef  = Ei - Eg

            J_p  = np.where(Eg > 0, Ji, Jf)
            B_p  = np.where(Eg > 0, Bif, Bfi)
            E_p  = np.where(Eg > 0, Ei, Ef) - Efond
            Eg_p = np.abs(Eg)

            all_Eg.append(Eg_p)
            all_B.append(B_p)
            all_Ep.append(E_p)

        Eg_all   = np.concatenate(all_Eg)
        B_all    = np.concatenate(all_B)
        Ep_all   = np.concatenate(all_Ep)
        Eexc_all = np.concatenate(all_Eexc)

        # BUILD RHO ON MERGED SPECTRUM
        Eexc_bins = np.arange(Eexc_all.min(), Eexc_all.max() + dE_talys, dE_talys)
        nEbin = len(Eexc_bins) - 1
        rho_merged = np.zeros(nEbin)
        for i in range(nEbin):
            cond = (Eexc_all >= Eexc_bins[i]) & (Eexc_all < Eexc_bins[i + 1])
            rho_merged[i] = np.sum(cond) / dE_talys

        # COMPUTE F
        Eg_bins = np.arange(Eg_all.min(), Eg_all.max() + dE_talys, dE_talys)
        nEg = len(Eg_bins) - 1
        F_sum   = np.zeros(nEg)
        count_F = np.zeros(nEg)

        for idx, (Eg_val, B_val, Ep_val) in enumerate(zip(Eg_all, B_all, Ep_all)):
            bin_E = np.searchsorted(Eexc_bins, Ep_val, side='right') - 1
            bin_E = np.clip(bin_E, 0, nEbin - 1)
            rho_val = rho_merged[bin_E]
            if rho_val == 0:
                continue
            f_val = C_TALYS * B_val * rho_val
            bin_Eg = np.searchsorted(Eg_bins, Eg_val, side='right') - 1
            if 0 <= bin_Eg < nEg:
                F_sum[bin_Eg]   += f_val
                count_F[bin_Eg] += 1

        F_talys = np.where(count_F > 0, F_sum / count_F, 0.0)
        Eg_centers = Eg_bins[:-1] + dE_talys / 2

        # WRITE TALYS .psf FILE
        element_symbol = self.current_element[:2]
        filename_psf = f"{element_symbol}.psf"
        trans_label  = f"f{self.transition.lower()}_CI-SM[mb/MeV]"
        with open(filename_psf, "w") as f:
            f.write(f" Z={self.Z:4d} A={self.A:4d}\n")
            f.write(f"    E[MeV]  {trans_label}\n")
            for E, fv in zip(Eg_centers, F_talys):
                f.write(f"    {E:.3f}   {fv:.3e}\n")
        print(f"\nTALYS file saved: {filename_psf}")

        # PLOT
        valid = count_F > 0
        plt.figure(figsize=(9, 5))
        plt.plot(Eg_centers[valid], F_talys[valid], drawstyle="steps-mid", linewidth=2)
        plt.yscale('log')
        plt.xlabel(r"$E_\gamma$ (MeV)", fontsize=12)
        plt.ylabel(f"$f_{{{self.transition}}}$  (mb/MeV)", fontsize=12)
        plt.title(f"Strength Function TALYS  —  {self.current_element}  (both parities)",
                  fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # OPTION 7 — Brink-Axel hypothesis check
    # ------------------------------------------------------------------

    def option_7_brink_axel(self):
        """
        Plot 1 — F by 2 MeV Eexc ranges + F total
        Plot 2 — F by spin contribution + F total
        """
        print("\n" + "-" * 50)
        print("OPTION 7: BRINK-AXEL HYPOTHESIS CHECK")
        print("-" * 50)

        if not self.data_loaded:
            print("Please load data first (option 9).")
            return

        all_spins = np.unique(self.J_parent)
        Emin_all  = self.E_parent_exc.min()
        Emax_all  = self.E_parent_exc.max()

        Eg_tot, count_tot, F_tot = self._compute_F(Emin_all, Emax_all, all_spins)
        if Eg_tot is None:
            return
        valid_tot = count_tot > 0

        # PLOT 1: F BY 2 MeV Eexc SLICES
        E_step   = 2.0
        E_ranges = []
        e = Emin_all
        while e < Emax_all:
            E_ranges.append((e, min(e + E_step, Emax_all)))
            e += E_step

        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(E_ranges)))

        for (Elow, Ehigh), col in zip(E_ranges, colors):
            Eg_c, cnt, F_c = self._compute_F(Elow, Ehigh, all_spins)
            if Eg_c is None:
                continue
            valid = cnt > 0
            if np.sum(valid) == 0:
                continue
            plt.plot(Eg_c[valid], F_c[valid], drawstyle="steps-mid",
                     linewidth=1.5, color=col, alpha=0.8,
                     label=f"$E_{{exc}}$=[{Elow:.0f},{Ehigh:.0f}] MeV")

        plt.plot(Eg_tot[valid_tot], F_tot[valid_tot], "k-", drawstyle="steps-mid",
                 linewidth=2.5, label="Total")
        plt.yscale('log')
        plt.xlabel(r"$E_\gamma$ (MeV)", fontsize=12)
        plt.ylabel(f"$f_{{{self.transition}}}$  (MeV$^{{-3}}$)", fontsize=12)
        plt.title(f"Brink-Axel — by $E_{{exc}}$ range  —  {self.current_element}", fontsize=12)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        # PLOT 2: F BY SPIN
        J_unique = np.unique(self.J_parent)
        plt.figure(figsize=(10, 6))
        colors2 = plt.cm.tab10(np.linspace(0, 1, len(J_unique)))

        for J_val, col in zip(J_unique, colors2):
            Eg_c, cnt, F_c = self._compute_F(Emin_all, Emax_all, [J_val])
            if Eg_c is None:
                continue
            valid = cnt > 0
            if np.sum(valid) == 0:
                continue
            plt.plot(Eg_c[valid], F_c[valid], drawstyle="steps-mid",
                     linewidth=1.5, color=col, alpha=0.8,
                     label=f"J={J_val/2:.1f}")   # display real J

        plt.plot(Eg_tot[valid_tot], F_tot[valid_tot], "k-", drawstyle="steps-mid",
                 linewidth=2.5, label="Total")
        plt.yscale('log')
        plt.xlabel(r"$E_\gamma$ (MeV)", fontsize=12)
        plt.ylabel(f"$f_{{{self.transition}}}$  (MeV$^{{-3}}$)", fontsize=12)
        plt.title(f"Brink-Axel — by spin  —  {self.current_element}", fontsize=12)
        plt.legend(fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        print("\n" + "=" * 50)
        print("      NUCLEAR PHYSICS PROGRAM")
        print("=" * 50)

        while True:
            self.menu_principal()
            choice = input("\nYour choice: ").strip()

            if choice == "1":
                self.option_1_level_density()
            elif choice == "2":
                self.option_2_spectrum()
            elif choice == "3":
                self.option_3_bethe()
            elif choice == "4":
                self.option_4_averages_B()
            elif choice == "5":
                self.option_5_strength_function()
            elif choice == "6":
                self.option_6_talys_format()
            elif choice == "7":
                self.option_7_brink_axel()
            elif choice == "9":
                self.data_load()
            elif choice == "0":
                print("\nGoodbye!")
                break
            else:
                print("Invalid option.")

            input("\nPress Enter to continue...")


if __name__ == "__main__":
    program = TIPP()
    program.run()
