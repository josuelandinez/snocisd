import numpy as np
from pyscf import gto, scf, ci, fci
np.set_printoptions(edgeitems=30, linewidth=100000, precision=5)
from noci_jax import nocisd
from noci_jax import slater
from noci_jax.misc import pyscf_helper


bl = 1.0
mol = gto.Mole()
mol.atom = f'''
H   0   0   0
H   0   0   {bl}
H   0   0   {2*bl}
H   0   0   {3*bl}
'''
mol.unit = "angstrom"
mol.basis = "sto3g"
mol.cart = True
mol.build()

mf = scf.UHF(mol)
mf.kernel()
# mo1 = mf.stability()[0]                                                             
# init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
# mf.kernel(init) 

h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=False)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron
myci = ci.UCISD(mf)
e_corr, civec = myci.kernel()
e_cisd = e_hf + e_corr 
c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)

#h1_mo, h2_mo = pyscf_helper.rotate_ham_spin0(mf)
#fcivec = myci.to_fcivec(civec, norb, nelec)
#print(fcivec)
#myfci = fci.FCI(mf)
#fcivec = np.loadtxt("./fcivec.txt")
#E = myfci.energy(h1_mo, h2_mo, fcivec, norb, nelec) + e_nuc
## print(E)
#e, v = myfci.kernel()
#print(v)
#exit()

dt = 0.1
ci_amps = nocisd.ucisd_amplitudes(myci, civec=civec, silent=True)
tmats, coeffs = nocisd.compress(ci_amps, dt1=dt, dt2=dt, tol2=1e-5)
nvir, nocc = tmats.shape[2:]
rmats = slater.tvecs_to_rmats(tmats, nvir, nocc)

E = slater.noci_energy(rmats, mo_coeff, h1e, h2e, return_mats=False, lc_coeffs=coeffs, e_nuc=e_nuc)
print(E)
