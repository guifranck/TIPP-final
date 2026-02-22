#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIPP Project: Computation of level densities and photon strength functions in atomic nuclei
Authors: Martí Serret and Guillaume Franck
Date: February 2026

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import math as m
import tkinter as tk
from scipy.interpolate import interp1d



###############################################################################
#           extraction des données des documents (densité de niveau)          #
###############################################################################

data_spec_te_neg = np.genfromtxt('te130neg_specj')
data_spec_te_pos = np.genfromtxt('te130pos_specj')

data_spec_xe_neg = np.genfromtxt('xe134neg_specj')
data_spec_xe_pos = np.genfromtxt('xe134pos_specj')

data_spec_sc_all = np.genfromtxt('sc44all_specj')

data_spec_ne_pos = np.genfromtxt('ne26pos_SPEC')



Jspin = [data_spec_te_neg[:,0] , data_spec_te_pos[:,0] , data_spec_xe_neg[:,0] , data_spec_xe_pos[:,0] , data_spec_sc_all[:,0] , data_spec_ne_pos[:,0]]
E_abs = [data_spec_te_neg[:,1] , data_spec_te_pos[:,1] , data_spec_xe_neg[:,1] , data_spec_xe_pos[:,1] , data_spec_sc_all[:,1] , data_spec_ne_pos[:,1]]



###############################################################################
#            extraction des données des documents (transitions)               #
###############################################################################


data_transM1_te_neg = np.genfromtxt('te130neg_m1')
data_transM1_te_pos = np.genfromtxt('te130pos_m1')

data_transM1_xe_neg = np.genfromtxt('xe134neg_m1')
data_transM1_xe_pos = np.genfromtxt('xe134pos_m1')

data_transM1_sc_all = np.genfromtxt('sc44e1')
data_trans_M1_ne_pos = np.genfromtxt('ne26pos_M1')


Jspin_i = [data_transM1_te_neg[:,0] , data_transM1_te_pos[:,0] , data_transM1_xe_neg[:,0], data_transM1_xe_pos[:,0] , data_transM1_sc_all[:,0] , data_trans_M1_ne_pos[:,0] ]

Jspin_f = [data_transM1_te_neg[:,1] , data_transM1_te_pos[:,1] , data_transM1_xe_neg[:,1], data_transM1_xe_pos[:,1] , data_transM1_sc_all[:,1] , data_trans_M1_ne_pos[:,1]]

E_i = [data_transM1_te_neg[:,2] , data_transM1_te_pos[:,2] , data_transM1_xe_neg[:,2], data_transM1_xe_pos[:,2] , data_transM1_sc_all[:,2] , data_trans_M1_ne_pos[:,2]]

E_gamma= [data_transM1_te_neg[:,3] , data_transM1_te_pos[:,3] , data_transM1_xe_neg[:,3], data_transM1_xe_pos[:,3] , data_transM1_sc_all[:,3] , data_trans_M1_ne_pos[:,3]]

Trans_if = [data_transM1_te_neg[:,4] , data_transM1_te_pos[:,4] , data_transM1_xe_neg[:,4], data_transM1_xe_pos[:,4] , data_transM1_sc_all[:,4] , data_trans_M1_ne_pos[:,4]]

Trans_fi = [data_transM1_te_neg[:,5] , data_transM1_te_pos[:,5] , data_transM1_xe_neg[:,5], data_transM1_xe_pos[:,5] , data_transM1_sc_all[:,5] , data_trans_M1_ne_pos[:,5] ]




labels =["te130 parity - ", "te130 parity +",  "xe134 parity -" , "xe134 parity +" , "sc44" , "ne26 parity +"]
labels_thalys = ["te130 ", "te130 ",  "xe134" , "xe134 " , "sc44" , "ne26 "]
units = ["MeV^{-3}" , "mb/MeV"]



###############################################################################
###############################################################################
#        creation de la fenetre permettant de faire les choix d'etude         #
###############################################################################
###############################################################################


def choix_parameter():

    root = tk.Tk()
    root.title('Study of the Photon Strength function: choice of study.')
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)
    root.geometry("800x600")
    root.minsize(800, 700)
   
    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Scrollbar verticale
    scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    content_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")


#Element    
    tk.Label(content_frame, text = 'Please choose the element you want to study').pack(anchor="w", padx=10, pady=(10, 0))
    element_var = tk.IntVar(value = 0)

    for i, e in enumerate(labels):
        tk.Radiobutton(
            content_frame,
            text=e,
            variable=element_var,
            value=i
        ).pack(anchor="w", padx=20)



#Bin
    tk.Label(content_frame, text = 'Please choose the Energy bin value you want to work with (in MeV)').pack(anchor="w", padx=10, pady=(10, 0))
    energy_var = tk.StringVar(value=.2)
    tk.Entry(content_frame, textvariable=energy_var, width=10).pack(anchor="w", padx=20)





#Unit
    tk.Label(content_frame, text = 'Please choose the system of unit you want to plot f in').pack(anchor="w", padx=10, pady=(10, 0))
    unit_var = tk.IntVar()
    for i, u in enumerate(units):
        tk.Radiobutton(
            content_frame,
            text=u,
            variable=unit_var,
            value=i
        ).pack(anchor="w", padx=20)
       
       
#E range

    tk.Label(content_frame, text = '').pack(anchor="w", padx=10, pady=(10, 0))  
   
    tk.Label(content_frame, text = 'The excitation energies are between the following values:').pack(anchor="w", padx=10, pady=(10, 0))
   
    tk.Label(content_frame, text = ' * Te130 parity - : [ 0.1948761499999705 ; 4.386612119999995 ] MeV').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Te130 parity + : [ 1.6475942100000225 ; 6.024887350000029 ] MeV').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Xe134 parity - : [ 0.2988258200000473 ; 4.740507420000029 ] MeV').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Xe134 parity + : [  1.6227045699999962 ; 6.072423740000033 ] MeV').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Sc44           : [  0.3035536200000024 ; 24.96106966000002 ] MeV').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Ne26 parity +  : [  3.6990152599999817 ; 11.002987459999986] MeV').pack(anchor="w", padx=10, pady=(10, 0))

    tk.Label(content_frame, text = '').pack(anchor="w", padx=10, pady=(10, 0))  

    tk.Label(content_frame, text = 'Please choose the minimal value you want to consider in excitation energy. If not, let it empty. (in MeV)').pack(anchor="w", padx=10, pady=(10, 0))  
    energy_var_min = tk.StringVar()
    tk.Entry(content_frame, textvariable = energy_var_min, width=10).pack(anchor="w", padx=20)

    tk.Label(content_frame, text = 'Please choose the maximal value you want to consider in excitation energy. If not, let it empty. (in MeV)').pack(anchor="w", padx=10, pady=(10, 0))
    energy_var_max = tk.StringVar()
    tk.Entry(content_frame, textvariable = energy_var_max, width=10).pack(anchor="w", padx=20)
       
   
 #J range  
    tk.Label(content_frame, text = '').pack(anchor="w", padx=10, pady=(10, 0))    
 
    tk.Label(content_frame, text = 'The spins  are between the following values:').pack(anchor="w", padx=10, pady=(10, 0))
     
    tk.Label(content_frame, text = ' * Te130 parity - : [ 0 ; 14 ] ').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Te130 parity + : [ 0 ; 14 ]').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Xe134 parity - : [ 0 ; 14 ]').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Xe134 parity + : [ 0 ; 14 ]').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Sc44           : [ 0 ; 24 ]').pack(anchor="w", padx=10, pady=(10, 0))
    tk.Label(content_frame, text = ' * Ne26 parity +  : [ 0 ; 12 ]').pack(anchor="w", padx=10, pady=(10, 0))

    tk.Label(content_frame, text = '').pack(anchor="w", padx=10, pady=(10, 0))  
   
   
    tk.Label(content_frame, text = 'Please choose the minimal value you want to consider in spin. If not, let it empty. ').pack(anchor="w", padx=10, pady=(10, 0))    
    J_var_min = tk.StringVar()
    tk.Entry(content_frame, textvariable = J_var_min, width=10).pack(anchor="w", padx=20)

    tk.Label(content_frame, text = 'Please choose the maximal value you want to consider in spin. If not, let it empty. ').pack(anchor="w", padx=10, pady=(10, 0))
    J_var_max = tk.StringVar()
    tk.Entry(content_frame, textvariable = J_var_max, width=10).pack(anchor="w", padx=20)
   
   
#density
    tk.Label(content_frame, text = 'Do you want to plot the density function  ?').pack(anchor="w", padx=10, pady=(10, 0))

    rho_var = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=rho_var, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=rho_var, value=True).pack(anchor="w", padx=20)
   
#levels
    tk.Label(content_frame, text = 'Do you want to plot the nuclear level spectrum ?').pack(anchor="w", padx=10, pady=(10, 0))

    level_var = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=level_var, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=level_var, value=True).pack(anchor="w", padx=20)
       
   
#Bethe
    tk.Label(content_frame, text = 'Do you want to plot the theoretical density function ?').pack(anchor="w", padx=10, pady=(10, 0))

    rho_theo_var = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=rho_theo_var, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=rho_theo_var, value=True).pack(anchor="w", padx=20)
   

   
#Transition
    tk.Label(content_frame, text = 'Do you want to plot the averaged value of transitions ? ').pack(anchor="w", padx=10, pady=(10, 0))

    transition = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=transition, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=transition, value=True).pack(anchor="w", padx=20)
       
#F
    tk.Label(content_frame, text = 'Do you want to plot the Strength function ? ').pack(anchor="w", padx=10, pady=(10, 0))

    f_var = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=f_var, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=f_var, value=True).pack(anchor="w", padx=20)

   
#THALYS
    tk.Label(content_frame, text = 'Do you want to export the strength function in THALYS format ? If yes, make sure to select the unit in mb/MEV.  ').pack(anchor="w", padx=10, pady=(10, 0))

    Thalys_var = tk.BooleanVar(value=False)
    tk.Radiobutton(content_frame, text="No",  variable=Thalys_var, value=False).pack(anchor="w", padx=20)
    tk.Radiobutton(content_frame, text="Yes", variable=Thalys_var, value=True).pack(anchor="w", padx=20)
       
    def force_unit_thalys(*args):
       if Thalys_var.get():
           unit_var.set(1)   # mb/MeV

    Thalys_var.trace_add("write", force_unit_thalys)
   

#Multiplot
    tk.Label(content_frame, text = 'Do you want to plot multiple elements on the same plot ? If not, let empty.').pack(anchor="w", padx=10, pady=(10, 0))

    element_var_multi= []
    for i, label in enumerate(labels):
        var = tk.BooleanVar(value = False)
        element_var_multi.append(var)
       
        tk.Checkbutton(
            content_frame,
            text = label,
            variable = var).pack(anchor="w" , padx = 20)
         
       

    result = []

    def valider():
        nonlocal result
        selected_el = [ i for i, var in enumerate(element_var_multi) if var.get()]
        result = {"element_index": element_var.get() ,
                  "energy_bin" : energy_var.get(),
                  "unit_index" : unit_var.get(),
                  "Emin": energy_var_min.get() ,
                  "Emax" : energy_var_max.get(),
                  "Jmin" : J_var_min.get(),
                  "Jmax" : J_var_max.get(),
                  "rho" : rho_var.get(),
                  "level" : level_var.get(),
                  "Bethe" : rho_theo_var.get(),
                  "trans" : transition.get(),
                  "f_plot" : f_var.get(),
                  "Thalys" : Thalys_var.get(),
                  "element_multi": selected_el}
       
       
        root.destroy()

    tk.Button(root , text = "Valider" , command = valider).pack(pady = 20)

    root.mainloop()

    return result


choix = choix_parameter()


print("Element :", labels[choix["element_index"]])
print(choix["element_index"])
print("Unit :", units[choix["unit_index"]])
print("Energy bin :", choix["energy_bin"])


print("Energy range :", choix["Emin"] ,choix["Emax"])
print("Spin range :", choix["Jmin"], choix["Jmax"])


element = choix["element_index"]
DE_bin =  float(choix["energy_bin"])
index_unit = choix["unit_index"]

nbr_multiplot = len(choix["element_multi"])
multi = choix["element_multi"]



###############################################################################
###############################################################################
#               Recherche de la Photon Strength function                      #
###############################################################################
###############################################################################


###############################################################################
#                           recherche du fondamental                          #
###############################################################################


E_fondamental = min((E_abs[element]))

E_max = max((E_abs[element]))
index_min = np.argmin(E_abs[element])
spin_fdmtl = Jspin[element][index_min]


print('The selected nucleus is:', labels[element])
print('Its fundamental level is a spin' , int(spin_fdmtl))
print('With energy ' , E_fondamental , " MeV ")

###############################################################################
#                                Num atomiq                                   #
###############################################################################


def num_atomiq():
    if element in [0,1] :
        A = 130
        Z= 52
        return A , Z
   
    if element in [2,3] :
        A = 134
        Z = 54
        return A , Z
    if element == 4 :
        A = 44
        Z = 21
        return A , Z
    if element == 5 :
        A = 26
        Z = 11
        return A , Z
       
###############################################################################
#                                 constante                                   #
###############################################################################


def constante(el = None):
    if el is None:
        el = element

    HBARC = 197.326
    e = np.sqrt(1.44)
   
    c_e = 16 * c.pi / 9 / (HBARC**3) * e**2
    c_m = 16 * c.pi * 0.105**2 / 9 / (HBARC**3) * e**2
   
    if index_unit == 0:
       
        if el == 4:
            cst = c_e
        else:
            cst = c_m
    else :
        cst = 8.674
    return cst
       

cst  = constante(element)


unit = 'MeV^-3'  if index_unit == 0 else 'mb/MeV'
trans = 'E1' if element == 4 else 'M1'
trans_thalys = 'e1' if element == 4 else 'e1'


###############################################################################
#                            etude critere                                    #
###############################################################################


def critere_etude(el = None):
    if el is None:
        el = element

    E_fondamental = min(E_abs[el])
    E_excitation= np.abs(E_abs[el]-E_fondamental)
    J = Jspin[el].astype(int)


    #Remise dans le bon sens des transitions (i->f):
    Spin_niveau = np.where(E_gamma[el] > 0 , Jspin_i[el] , Jspin_f[el])
    Transition_niveau = np.where(E_gamma[el] > 0 , Trans_if[el] , Trans_fi[el])
    E_f = E_i[el] - E_gamma[el]
    Energy_niveau = np.where(E_gamma[el] > 0 , E_i[el] , E_f )
    E_exc_niv =  Energy_niveau - E_fondamental
    Eg = abs(E_gamma[el])


    Emin_input = float(choix["Emin"]) if choix["Emin"] != "" and choix["Emin"] is ValueError else None
    Emax_input = float(choix["Emax"]) if choix["Emax"] != "" and choix["Emax"] is ValueError else None


    Emin_exc = min(E_excitation)
    Emax_exc = max(E_excitation)


    a = Emin_input if (Emin_input is not None and Emin_input > Emin_exc) else Emin_exc
    b = Emax_input if (Emax_input is not None and Emax_input < Emax_exc) else Emax_exc

   
    Jmin_input = float(choix["Jmin"]) if choix["Jmin"] != "" and choix["Jmin"] is ValueError else None
    Jmax_input = float(choix["Jmax"]) if choix["Jmax"] != "" and choix["Jmax"] is ValueError else None

    Jmin_exc = min(Spin_niveau)
    Jmax_exc = max(Spin_niveau)


    Ja = Jmin_input if (Jmin_input is not None and Jmin_input > Jmin_exc) else Jmin_exc
    Jb = Jmax_input if (Jmax_input is not None and Jmax_input < Jmax_exc) else Jmax_exc


    mask_E = (E_exc_niv >= a) & ( b >= E_exc_niv)
    mask_J =(Spin_niveau >= Ja) & ( Jb >= Spin_niveau)
    mask =  mask_E & mask_J

    E_exc_accepted = E_exc_niv[mask]
    J_exc_accepted = Spin_niveau[mask]
    Trans_accepted = Transition_niveau[mask]
    Eg_accepted = Eg[mask]

    return {
        "E_fondamental" : E_fondamental,
        "E_excitation" : E_excitation,
        "J" : J,
        "Spin_niveau" : Spin_niveau,
        "Trans_niveau" : Transition_niveau,
        "E_exc_niv" : E_exc_niv,
        "Eg" : Eg,
        "a" : a,
        "b" : b,
        "Ja" : Ja,
        "Jb" : Jb,
        "mask" : mask,
        "E_exc_accepted": E_exc_accepted,
        "J_exc_accepted": J_exc_accepted,
        "Trans_accepted": Trans_accepted,
        "Eg_accepted" : Eg_accepted,
    }



###############################################################################
#                                  spectre                                    #
###############################################################################


def spectre():
    E_excitation= np.abs(E_abs[element]-E_fondamental)

    fig, ax = plt.subplots(figsize=(3, 6))

    x0, x1 = 0.3,0.6
    for E in E_excitation:
        ax.hlines(E, x0, x1, linewidth=1)

    ax.set_ylim(min(E_excitation) - 0.5, max(E_excitation) + 0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Excitation energy (MeV)")
    plt.title(f"Spectrum for {labels[element]}")
    plt.show()

if choix["level"] == True:
    spectre()



###############################################################################
#                fonction de densité de niveau par spin                       #
###############################################################################



def fct_rho_j(el = None):
    if el is None:
        el = element
       
    crit = critere_etude(el)
   
    a = crit["a"]
    b  = crit["b"]
    Ja = crit["Ja"]
    Jb = crit["Jb"]
    J = crit["J"]
    E_excitation = crit["E_excitation"]
   
    E_bins_ok = np.arange(a ,b + DE_bin , DE_bin)
    nE = len(E_bins_ok) - 1


    J_values_fixed = np.arange(Ja, Jb + 2, 2)
    nJ = len(J_values_fixed)
   
    rho_j = np.zeros((nE, nJ))
    count = np.zeros((nE, nJ))

    for i in range(nE):
   
        for j, Jval in enumerate(J_values_fixed):

            count[i,j] = np.sum((E_excitation >= E_bins_ok[i]) & (E_excitation < E_bins_ok[i+1]) & (J == Jval))
            rho_j[i, j] = count[i,j] / DE_bin  


    bin_c = 0.5 * (E_bins_ok[:-1] + E_bins_ok[1:])
   
    print("-" * 50)
    print('The value of the density level is given by the following matrix. Each entry corresponds to the partial density (per Energy per spin):')
    print(rho_j)
    print("-" * 50)

    return bin_c , nE, rho_j

#################
#mise en fichier#
#################

x_r , ne , y_r_j = fct_rho_j()
y_r = np.sum(y_r_j , axis = 1)
fichier_rho = np.column_stack((x_r,y_r))
nom_rho = f"value_rho_J_{labels[element]}.txt"


np.savetxt(nom_rho, fichier_rho, header=r"$E_{exc}$  $\rho_{Ebin}$", fmt='%.2e')

######
#plot#
######
 
if choix["rho"] == True:

    x_rho ,ne, y_rho = fct_rho_j()
    y_tot = np.sum(y_rho,axis=1)
    plt.figure()
    plt.title(f"Density function plotted for a {trans}\n transition for {labels[element]}\n")
    plt.step(x_rho , y_tot ,where ='mid', label = labels[element])
    plt.xlabel("$E_{excitation}$ (MeV)")
    plt.ylabel(r"$\rho$ (MeV$^{-1}$)")
    plt.yscale('log')
    plt.legend()
    plt.show()

   
       

###############################################################################
#                                 Bethe formula                               #
###############################################################################

def fct_rho_bethe(el = None):
    if el is None:
        el = element
   
    crit = critere_etude(el)

    a = crit["a"]
    b  = crit["b"]

    A , Z = num_atomiq()
    a_bethe = A/10
    fac = np.sqrt(m.pi) / 12 / a_bethe**(-1/4)
   
    E_bethe = np.linspace(a , b , num=256)
   
    rho_theo =  fac * np.exp(2*np.sqrt(a_bethe*E_bethe[E_bethe > 0])) / (E_bethe[E_bethe > 0])**(5/4)
       
    return E_bethe[E_bethe > 0] , rho_theo

E_center_beth , rho_beth = fct_rho_bethe()

######
#plot#
######

if choix["Bethe"] == True:

    plt.figure()
    plt.title("Theoretical density function using the Bethe-formula")
    plt.step(E_center_beth,rho_beth,where ='mid', label = labels[element])
    plt.xlabel("$E_{excitation}$ (MeV)")
    plt.ylabel(r"$\rho$ $_{bethe}$ (MeV$^{-1}$)")
    plt.yscale('log')
    plt.legend()
    plt.show()



###############################################################################
#                            fonction B par spin                              #
###############################################################################

def fct_B_j(el=None):
    if el is None:
        el = element

    crit = critere_etude(el)
    Eg = crit["Eg"]
    Trans_tot = np.where(E_gamma[el] > 0, Trans_if[el], Trans_fi[el])

    Eming = min(Eg)
    Emaxg = max(Eg)
    E_binsg = np.arange(Eming, Emaxg + DE_bin, DE_bin)
    nEg = len(E_binsg) - 1

    B = np.zeros(nEg)
    count = np.zeros(nEg)

    for j in range(nEg):
        for i in range(len(Trans_tot)):
            if (Eg[i] >= E_binsg[j]) & (Eg[i] < E_binsg[j+1]):
                count[j] += 1
                B[j]     += Trans_tot[i]

    B_norm = np.zeros(nEg)
   
    for j in range(nEg):
        if count[j] != 0:
            B_norm[j] = B[j] / count[j]
        else:
            B_norm[j] = 0

    bin_centers_B = 0.5 * (E_binsg[:-1] + E_binsg[1:])

    return bin_centers_B, B_norm


x_b  , y_b = fct_B_j()

##################
#Mise en fichiers#
##################

fichier_B = np.column_stack((x_b,y_b))
nom_rB = f"value_B_J_{labels[element]}.txt"
formats_B = '%.3e'


if element == '4':
    np.savetxt(nom_rB, fichier_B, header = '$E_{\gamma}$  $<B(E1)>_{Ebin}$', fmt =formats_B)
else :
    np.savetxt(nom_rB, fichier_B, header = '$E_{\gamma}$  $<B(M1)>_{Ebin}$', fmt =formats_B)

######
#Plot#
######

if choix["trans"] == True:
    x_trans , y_trans = fct_B_j()
   
    if element == 4:
        ylab = (r"$<B(E1)>$ $(\mu_N)^2$")
    else:
        ylab = (r"$<B(M1)>$ $(\mu_N)^2$")
   
    plt.figure()
    plt.title(f"Transition function for a {trans}\n transition")
    plt.step(x_trans,y_trans ,where ='mid', label = labels[element])
    plt.xlabel("$E_{gamma}$ (MeV)")
    plt.ylabel(ylab)
    plt.yscale('log')
    plt.legend()
    plt.show()
   


###############################################################################
#                                  f par spin                                 #
###############################################################################

def fct_f_j(el = None):
    if el is None:
        el = element
       
    crit = critere_etude(el)

     # Extraction des variables depuis le dictionnaire
    a = crit["a"]
    Ja = crit["Ja"]
    Eg = crit["Eg"]
    E_exc_accepted = crit["E_exc_accepted"]
    J_exc_accepted = crit["J_exc_accepted"]
    Trans_accepted = crit["Trans_accepted"]
    Eg_accepted    = crit["Eg_accepted"]
       

    bins , nE , rho_j = fct_rho_j(el)
   
    #Binning sur les energies gamma:

    Eg_max = max(Eg)
    Eg_min = min(Eg)

    Ebins_g = np.arange(Eg_min , Eg_max + DE_bin , DE_bin)
    nbr_bins_g = len(Ebins_g) - 1
   

    #Determination des bins pour ranger F. On parcours les valeurs possibles du fichier Transitions

    F_unwrapped = np.zeros(len(E_exc_accepted))
    F = np.zeros(nbr_bins_g)
    countF = np.zeros(nbr_bins_g)

    for i in range(len(E_exc_accepted)):
       
        E_exc = E_exc_accepted[i]
        J_exc = J_exc_accepted[i]
       
        #On fait le lien entre les niveaux d'énergies de transitions et ceux du fichier spec
        Trans_exc = Trans_accepted[i]


        bin_E = int((E_exc - a) / DE_bin)
        if bin_E >= nE:
            bin_E = nE - 1
       
           
        bin_J = int((J_exc - Ja) /  2)

        F_unwrapped[i] = constante(el) * Trans_exc * rho_j[bin_E , bin_J]


    #Il faut maintenant ranger dans les bons bins:

    for j in range(nbr_bins_g):
        for i in range(len(F_unwrapped)):
            if ((Eg_accepted[i] >= Ebins_g[j]) & (Eg_accepted[i] < Ebins_g[j+1])):
                countF[j] +=1
                F[j] += F_unwrapped[i]


    #Normalisation:
    F_tot = np.zeros(nbr_bins_g)
    for i in range(nbr_bins_g):
        if countF[i] != 0 :
            F_tot[i] = F[i] / countF[i]
        else:
            F_tot[i] = 0

    Eg_centers = 0.5 * (Ebins_g[:-1] + Ebins_g[1:])
   
    print(F_tot)

    return Eg_centers , F_tot



x_f , y_f = fct_f_j()



###############################################################################
#                             Export des données                              #
###############################################################################

#####################
#moyenne pour thalys#
#####################

def fct_f_moyenne():
    paires = {0:1, 1:0, 2:3, 3:2}
    if element not in paires:
        return x_f, y_f

    x1, y1 = fct_f_j(element)
    x2, y2 = fct_f_j(paires[element])

    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    x_commun = np.linspace(x_min, x_max, 256)

    #grille commune pour faire la moyenne
    f1 = interp1d(x1, y1, bounds_error=False, fill_value=0.0)
    f2 = interp1d(x2, y2, bounds_error=False, fill_value=0.0)

    y_moyenne = (f1(x_commun) + f2(x_commun)) / 2.0

    return x_commun, y_moyenne

#############
#export de f#
#############

def fichier_valeurs_f(el = None):
    if el is None:
        el = element
       
    if choix["Thalys"] == False:
        if int(choix["unit_index"]) == 0:
           
            fichier_val = np.column_stack((x_f,y_f))
            nom_fichier = f"value_f_J_{labels[el]}.txt"
            formats = ['%.2e', '%.5e']

            print(f"The datas are going to be stored in a .txt file named : {nom_fichier}")

            if el == 4:
                np.savetxt(nom_fichier, fichier_val, header = '$E_{\gamma}$  $<f(E1)> in MeV^-3$', fmt =formats)
            else :
                np.savetxt(nom_fichier, fichier_val, header = '$E_{\gamma}$  $<f(M1)> in MeV^-3$', fmt =formats)
               
        if int(choix["unit_index"]) == 1:
           
            fichier_val = np.column_stack((x_f,y_f))
            nom_fichier = f"AA.value_f_J_{labels[el]}.txt"
            formats = ['%.2e', '%.5e']

            print(f"The datas are going to be stored in a .txt file named : {nom_fichier}")

            if el == 4:
                np.savetxt(nom_fichier, fichier_val, header = '$E_{\gamma}$  $<f(E1)> in mb/MeV $', fmt =formats)
            else :
                np.savetxt(nom_fichier, fichier_val, header = '$E_{\gamma}$  $<f(M1)> in mb/MeV$', fmt =formats)
   
    #Export en configuration thalys:
       
    else:
        A, Z = num_atomiq()
       
        x_thalys , y_thalys = fct_f_moyenne()
       
        fichier_val = np.column_stack((x_thalys, y_thalys))
        nom_fichier = f"{labels_thalys[el]}.psf"

        with open(nom_fichier, 'w') as f:
            # Première ligne : Z et A
            f.write(f" Z = {Z}  A = {A}\n")
            # Deuxième ligne : noms de colonnes
            f.write(f"E[MeV]  f{trans_thalys}_CI-SM[{unit}]\n")
            # Ensuite les données
            for row in fichier_val:
                f.write(f"{row[0]:.3e}  {row[1]:.3e}\n")

        print(f"The datas are going to be stored in a .psf file named : {nom_fichier}")
       

fichier_valeurs_f()


###############################################################################
###############################################################################
#                                                                             #
#                                   PLOTS                                     #
#                                                                             #
###############################################################################
###############################################################################


def multiple_plot():
       
    plt.figure()

    for elmt in choix["element_multi"]:
            y_label = f"$f_{{X1}}$ ({unit})"
            x , y = fct_f_j(elmt)
            plt.step(x,y,where ='mid', label = labels[elmt])
    plt.xlabel("$E_{\gamma}$ (MeV)")
    plt.xlim(0,6)
    plt.ylabel(y_label)
    plt.ylim(10e-11 , 10e-7)
    plt.yscale('log')
    plt.legend()
    plt.show()
           
   


###############################################################################
#                                     plot de f                               #
###############################################################################



def plot_f():
    x_trac , y_trac = fct_f_j()
    y_label = f"$f({trans})$ ({unit})"
    if element in [0,1,2,3,5]:
        plt.figure()
        plt.title(f"plot of the dipole strength function for the element {labels[element]} for a {trans} transition\n")
        plt.step(x_trac , y_trac ,where ='mid', label = labels[element])
        plt.xlabel("$E_{\gamma}$ (MeV)")
        plt.xlim(0,6)
        plt.ylabel(y_label)
        plt.yscale('log')
        plt.legend()
        plt.show()
    else :
        plt.figure()
        plt.title(f"plot of the dipole strength function for the element {labels[element]} for a {trans} transition\n")
        plt.step(x_f,y_f,where ='mid', label = labels[element])
        plt.xlabel("$E_{\gamma}$ (MeV)")
        plt.xlim(0,6)
        plt.ylabel(y_label)
        plt.yscale('log')
        plt.legend()
        plt.show()
       
if choix["f_plot"] == True:
    plot_f()
   
   

###############################################################################
#                               multiple plot                                 #
###############################################################################

if choix["element_multi"] != []:
    multiple_plot()
