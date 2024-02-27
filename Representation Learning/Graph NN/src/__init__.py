import logging
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*value needs to be floating point.*")
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")
warnings.filterwarnings("ignore", ".*The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.*")
