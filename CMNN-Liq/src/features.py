"""CPT inputs; pga is PGA/g (numerically PGA in g), not m/s^2."""
import numpy as np
SOIL = ["mw", "depth", "stress_ratio", "qt1n", "ic"]

def physical_features(d):
    z = np.asarray(d['depth'], float); r = np.asarray(d['stress_ratio'], float)
    ic = np.asarray(d['ic'], float); q = np.asarray(d['qt1n'], float)
    m = np.asarray(d['mw'], float); a = np.asarray(d['pga'], float)
    if not np.isfinite(np.column_stack([z,r,ic,q,m,a])).all() or np.any(np.column_stack([z,r,ic,q,m,a]) <= 0) or np.any(z >= 40):
        raise ValueError('Invalid frozen-model inputs')
    rd = np.where(z > 9.15, 1.174-.0267*z, 1-.00765*z)
    sig_eff = 19*z/r; t = sig_eff/100
    ca = np.minimum(1.4314, -.016*t**3+.178*t**2-.063*t+.903)
    kc = 2.429*ic**4-16.943*ic**3+44.551*ic**2-51.497*ic+22.802
    qcs = np.minimum(175, q*kc)
    crr = ca*np.exp(-2.957+1.264*(qcs/100)**1.25)
    msf = (m/7.5)**-2.56
    fsref = crr/(.65*.1*r*rd/msf)
    if not np.isfinite(fsref).all() or np.any(fsref <= 0):
        raise ValueError('Nonpositive/nonfinite reference FS')
    return dict(rd=rd,sigma_effective_est_kpa=sig_eff,ca=ca,kc=kc,qcs=qcs,crr=crr,msf=msf,fs_ref=fsref,fs_current=fsref*.1/a)

def inputs(d):
    f = physical_features(d)
    return np.column_stack([*[np.asarray(d[k],float) for k in SOIL],np.log(f['fs_ref']),np.log(np.asarray(d['pga'],float)/.1)])
def juang(d, legacy_vector=False):
    """Author equations, with rowwise rd fix; stress estimated using 19 kN/m3."""
    depth=d.depth.to_numpy(float); s=d.stress_ratio.to_numpy(float)
    rd=np.where(depth>9.15,1.174-.0267*depth,1-.00765*depth)
    if legacy_vector:
        rd=1.174-.0267*depth if np.all(depth>9.15) else 1-.00765*depth
    sig=19*depth/s; t=sig/100
    ca=np.minimum(1.4314,-.016*t**3+.178*t**2-.063*t+.903)
    ic=d.ic.to_numpy(float)
    kc=2.429*ic**4-16.943*ic**3+44.551*ic**2-51.497*ic+22.802
    qcs=np.minimum(175,d.qt1n.to_numpy(float)*kc)
    crr=ca*np.exp(-2.957+1.264*(qcs/100)**1.25)
    msf=(d.mw.to_numpy(float)/7.5)**-2.56
    csr=.65*d.pga.to_numpy(float)*s*rd/msf
    fs=crr/csr
    if not np.all(np.isfinite(fs)&(fs>0)):
        raise ValueError('Nonpositive/nonfinite FS: inspect input/domain; never silently clip.')
    return fs
def physical_reference(d):
    fixed=d.copy();fixed['pga']=.1
    return juang(fixed)

def validate(d):
    v=d[SOIL+['pga']].to_numpy(float)
    if not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Missing, nonfinite or nonpositive required input')
    if np.any(d.depth>=40):raise ValueError('Depth outside supported shallow-soil domain (<40m)')
