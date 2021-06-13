import numpy as np
import scipy.stats
from scipy.special import gamma

# Contains the function used in Kingman_Bessel_Sampling.ipynb, sampling Kingman distributions from Bessel process's excursion lengths around 0 (details in Pitman-Yor paper)

def explicit_bessel(delta, n):
  """
    Bessel process sampling between time [0,1], dimension delta, n points
    Using explicit Euler scheme
  """
  traj = np.zeros(n)
  w = np.random.normal(scale=1/np.sqrt(n), size=n)
  for i in range(1,n):
    traj[i] = traj[i-1] + delta/n + 2*np.sqrt(abs(traj[i-1]))*w[i-1]
  return np.sqrt(abs(traj))


def implicit_bessel(delta, n):
  """
    Bessel process sampling between time [0,1], dimension delta, n points
    Using explicit Euler scheme
  """
  traj = np.zeros(n)
  w = np.random.normal(scale=1/np.sqrt(n), size=n)
  for i in range(1,n):
    traj[i] = (w[i] + np.sqrt(w[i]**2 + delta/n + traj[i-1]))**2
  return np.sqrt(abs(traj))


def excursion_lengths(traj, thr=1e-1):
  """
  Take a Bessel process trajectory in input, an array of shape [n]
  Output an array of shape [n], with at index i the length of the i_th longest excursion around 0 o the trajectory
  """
  n = traj.shape[0]
  len_distrib = np.zeros(n)
  zeros = traj<thr
  zeros_ind = (np.cumsum(np.ones(n))-1)[zeros]
  lengths = zeros_ind[1:] - zeros_ind[:-1]
  for i in range(len(lengths)):
    len_distrib[i] = int(lengths[i])
  return -np.sort(-len_distrib)


def brownian_excursion_lengths(traj):
  """
  Take a Brownian motion trajectory in input, an array of shape [n]
  Output an array of shape [n], with at index i the length of the i_th longest excursion around 0 o the trajectory
  """
  n = traj.shape[0]
  len_distrib = np.zeros(n)
  zeros = np.append(traj[1:]*traj[:-1]<=0, False)
  zeros_ind = (np.cumsum(np.ones(n))-1)[zeros]
  lengths = zeros_ind[1:] - zeros_ind[:-1]
  for i in range(len(lengths)):
    len_distrib[i] = int(lengths[i])
  return -np.sort(-len_distrib)


def brownian_bridge(n):
  """
    Output a Brownian bridge sampling between time [0,1], dimension delta, n points, initial point B_0=0 and final point B_1=0
    With Z_t Brownian motion, use the fact that Z_t - t*Z_1 is a Brownian bridge
  """
  w = np.cumsum(np.random.normal(scale=1/np.sqrt(n), size=n))
  t = np.arange(n)/(n-1)
  return w - t*w[-1]