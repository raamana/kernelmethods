from abc import ABC, abstractmethod
from collections import Iterable
from copy import copy
from itertools import product as iter_product

import numpy as np
from scipy.sparse import issparse, lil_matrix

from kernelmethods.operations import center_km, is_PSD, normalize_km, \
    normalize_km_2sample, frobenius_norm
from kernelmethods.utils import check_callable, ensure_ndarray_1D, ensure_ndarray_2D, \
    get_callable_name, not_symmetric
from kernelmethods import config as cfg

class KernelMethodsException(Exception):
    """
    Generic exception to indicate invalid use of the kernelmethods library.


    Allows to distinguish improper use of KernelMatrix from other code exceptions
    """
    pass


class KMAccessError(KernelMethodsException):
    """Exception to indicate invalid access to the kernel matrix!"""
    pass


class KMSetAdditionError(KernelMethodsException):
    """Exception to indicate invalid addition of kernel matrix to a KernelSet"""
    pass


class BaseKernelFunction(ABC):
    """
    Abstract base class for kernel functions.

    Enforces each derived kernel:
    1. to be callable, with two inputs
    2. to have a name and a str representation
    3. provides a method to check whether the derived kernel func is a valid kernel
       i.e. the kernel matrix derived on a random sample is positive semi-definite (PSD)
    4. and that it is symmetric (via tests) as required.

    """


    def __init__(self, name):
        """
        Constructor.

        Parameters
        ----------
        name : str
            short name to describe the nature of the kernel function

        """

        self.name = name


    @abstractmethod
    def __call__(self, x, y):
        """Actual computation to defined in the inherited class!"""


    def is_psd(self):
        """Tests whether kernel matrix produced via this function is PSD"""

        # passing the instance of the derived class
        km = KernelMatrix(self)

        km.attach_to(np.random.rand(50, 4))  # random_sample
        return is_PSD(km.full)


    @abstractmethod
    def __str__(self):
        """Representation"""


    # aliasing others to __str__ for now
    def __format__(self, _):
        """Representation"""

        return self.__str__()


    def __repr__(self):
        """Representation"""

        return self.__str__()


class KernelFromCallable(BaseKernelFunction):
    """Class to create a custom kernel from a given callable."""


    def __init__(self, input_func, name=None, **func_params):
        """
        Constructor.

        Parameters
        ----------
        input_func : callable
            A callable that can accept atleast 2 args
            Must not be builtin or C function.
            If func is a C or builtin func, wrap it in a python def

        name : str
            A name to identify this kernel in a human readable way

        func_params : dict
            Parameters to func

        """

        self.func = check_callable(input_func, min_num_args=2)
        self.params = func_params

        super().__init__(name=get_callable_name(input_func, name))


    def __call__(self, x, y):
        """Actual computation!"""

        return self.func(x, y, **self.params)


    def __str__(self):
        """human readable repr"""

        arg_repr = '({})'.format(self.params) if len(self.params) > 0 else ''
        return "{}{}".format(self.name, arg_repr)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class KernelMatrix(object):
    """
    A self-contained class for the Gram matrix induced by a kernel function on a
    sample.

    KM[i,j] --> kernel between samples i and j
    KM[set_i,set_j] where len(set_i)=m and len(set_i)=n --> matrix KM of size mxn
        where KM_ij = kernel between samples set_i(i) and set_j(j)

    """


    def __init__(self,
                 kernel,
                 normalized=True,
                 name='KernelMatrix'):
        """
        Constructor.

        Parameters
        ----------
        kernel : BaseKernelFunction
            kernel function that populates the kernel matrix

        normalized : bool
            Flag to indicate whether to normalize the kernel matrix
            Normalization is recommended, unless you have clear reasons not to.

        name : str
            short name to describe the nature of the kernel function

        """

        if not isinstance(kernel, BaseKernelFunction):
            raise TypeError('Input kernel must be derived from '
                            ' kernelmethods.BaseKernelFunction')

        if not isinstance(normalized, bool):
            raise TypeError('normalized flag must be True or False')

        self.kernel = kernel
        self._keep_normed = normalized
        self.name = name

        # to ensure we can always query the size attribute
        self._num_samples = None
        self._sample = None
        self._sample_name = None

        # user-defined attribute dictionary
        self._attr = dict()

        self._reset()


    def attach_to(self,
                  sample_one,
                  name_one='sample',
                  sample_two=None,
                  name_two=None):
        """
        Attach this kernel to a given sample.

        Any computations from previous samples and their results will be reset,
            along with all the previously set attributes.

        Parameters
        ----------
        sample_one : ndarray
            Input sample to operate on
            Must be a 2D dataset of shape (num_samples, num_features)
                e.g. MLDataset or ndarray
            When sample_two=None (e.g. during training), sample_two refers to sample_one.

        name_one : str
            Name for the first sample.

        sample_two : ndarray
            Second sample for the kernel matrix i.e. Y in K(X,Y)
            Must be a 2D dataset of shape (num_samples, num_features)
                e.g. MLDataset or ndarray
            The dimensionality of this sample (number of columns, sample_two.shape[1])
                must match with that of sample_one

        name_two : str
            Name for the second sample.
        """

        self._sample = ensure_ndarray_2D(sample_one, ensure_dtype=sample_one.dtype)
        self._sample_name = name_one

        if sample_two is None:
            self._sample_two = self._sample
            self._name_two = name_one

            self._num_samples = self._sample.shape[0]
            self.shape = (self._num_samples, self._num_samples)
            self._two_samples = False

            self._sample_descr = "{} {}".format(self._sample_name, self._sample.shape)

        else:
            self._sample_two = ensure_ndarray_2D(sample_two,
                                                 ensure_dtype=sample_two.dtype)

            if self._sample.shape[1] != self._sample_two.shape[1]:
                raise ValueError('Dimensionalities of the two samples differ!')

            self._name_two = name_two
            self._num_samples = (self._sample.shape[0], self._sample_two.shape[0])
            self.shape = (self._sample.shape[0], self._sample_two.shape[0])

            self._two_samples = True

            self._sample_descr = "{} {} x {} {}" \
                                 "".format(self._sample_name, self._sample.shape,
                                           self._name_two, self._sample_two.shape)

        # cleanup old flags and reset to ensure fresh slate for this sample
        self._reset()


    def set_attr(self, name, value):
        """
        Sets user-defined attributes for the kernel matrix.

        Useful to identify this kernel matrix in various aspects!
        You could think of them as tags or identifiers etc.
        As they are user-defined, they are ideal to represent user needs and applications.
        """

        self._attr[name] = value


    def get_attr(self, attr_name, value_if_not_found=None):
        """Returns the value of an user-defined attribute.

        If not set previously, or no match found, returns value_if_not_found.
        """

        return self._attr.get(attr_name, value_if_not_found)


    def attributes(self):
        """
        Returns all the attributes currently set.
        """

        return self._attr


    @property  # this is to prevent accidental change of value
    def num_samples(self):
        """
        Returns the number of samples in the sample this kernel is attached to.

        This would be a scalar when the current instance is attached to a single sample.
        When a product of two samples i.e. K(X,Y) instead of K(X,X), it is an array of 2
        scalars representing num_samples from those two samples.
        """

        return self._num_samples


    def _reset(self):
        """Convenience routine to reset internal state"""

        self._populated_fully = False
        self._lower_tri_km_filled = False
        if hasattr(self, '_full_km'):
            delattr(self, '_full_km')
        self._is_centered = False
        self._is_normed = False

        # As K(i,j) is the same as K(j,i), only one of them needs to be computed!
        #  so internally we could store both K(i,j) and K(j,i) as K(min(i,j), max(i,j))
        self._KM = dict()

        # restricting attributes to the latest sample only, to avoid leakage!!
        self._attr.clear()

        # debugging and efficiency measurement purposes
        # for a given sample (of size n),
        #   number of kernel evals must never be more than n+ n*(n-1)/2 (or n(n+1)/2)
        #   regardless of the number of times different forms of KM are accessed!
        self._num_ker_eval = 0


    @property
    def size(self):
        """
        Returns the size of the KernelMatrix (total number of elements)
            i.e. num_samples from which the kernel matrix is computed from.
            In a single-sample case, it is the num_samples in the dataset.
            In two-sample case, it is the product of num_samples from two datasets.

        Defining this to correspond to .size attr of numpy arrays
        """

        if not self._two_samples:
            return self._num_samples**2
        else:
            return np.prod(self._num_samples)


    def __len__(self):
        """Convenience wrapper for .size attribute, to enable use of len(KernelMatrix)"""

        return self.size


    @property
    def full(self):
        """Fully populated kernel matrix in dense ndarray format."""

        if not self._populated_fully:
            self._populate_fully(fill_lower_tri=True, dense_fmt=True)

        if self._keep_normed:
            if not self._is_normed:
                self.normalize()
            return self._normed_km
        else:
            return self._full_km


    @property
    def full_sparse(self):
        """Kernel matrix populated in upper tri in sparse array format."""

        return self._populate_fully(fill_lower_tri=False)


    def center(self):
        """Method to center the kernel matrix"""

        if self._two_samples:
            raise NotImplementedError('Centering is not implemented (or possible)'
                                      ' when KM is attached two separate samples.')

        if not self._populated_fully:
            self._full_km = self._populate_fully(fill_lower_tri=True, dense_fmt=True)

        self._centered = center_km(self._full_km)
        self._is_centered = True


    def normalize(self, method='cosine'):
        """
        Normalize the kernel matrix to have unit diagonal.

        Cosine normalization mplements definition according to Section 5.1 in
            Shawe-Taylor and Cristianini, "Kernels Methods for Pattern Analysis", 2004

        """

        if not self._populated_fully:
            self._populate_fully(dense_fmt=True, fill_lower_tri=True)

        if not self._is_normed:
            if not self._two_samples:
                self._normed_km = normalize_km(self._full_km, method=method)
            else:
                # KM_XX and KM_YY must NOT be normalized for correct norm of K_XY
                #   NOTE: K_XY may NOT have unit diagonal
                #       as k(x,y) != sqrt(k(x,x))*sqrt(k(y,y))
                KM_XX = KernelMatrix(self.kernel, normalized=False)
                KM_XX.attach_to(sample_one=self._sample)

                KM_YY = KernelMatrix(self.kernel, normalized=False)
                KM_YY.attach_to(sample_one=self._sample_two)

                # not passing .full_km for KM_XX and KM_YY as we only need their diagonal
                self._normed_km = normalize_km_2sample(self._full_km,
                                                       KM_XX.diagonal(),
                                                       KM_YY.diagonal())
            self._is_normed = True


    @property
    def centered(self):
        """Exposes the centered version of the kernel matrix"""

        if self._two_samples:
            raise KMAccessError('Centering not defined when attached to 2 samples!')

        if not self._is_centered:
            self.center()

        return self._centered


    @property
    def frob_norm(self):
        """Returns the Frobenius norm of the current kernel matrix"""

        if not self._populated_fully:
            self._populate_fully(dense_fmt=True, fill_lower_tri=True)

        if not hasattr(self, '_frob_norm'):
            self._frob_norm = frobenius_norm(self._full_km)

        return self._frob_norm


    def diagonal(self):
        """
        Returns the diagonal of the kernel matrix, when attached to a single sample.

        Raises
        ------
            ValueError
                When this instance is attached to more than one sample
        """

        if self._two_samples:
            raise KMAccessError('Diagonal() not defined when attached to 2 samples!')

        return np.array([self._eval_kernel(idx, idx) for idx in range(self.shape[0])])


    @property
    def normed_km(self):
        """Access to the normalized kernel matrix."""

        if not self._is_normed:
            self.normalize()

        return self._normed_km


    def _eval_kernel(self, idx_one, idx_two):
        """Returns kernel value between samples identified by indices one and two"""

        # maintaining only upper triangular parts, when attached to a single sample
        #   by ensuring the first index is always <= second index
        if idx_one > idx_two and not self._two_samples:
            idx_one, idx_two = idx_two, idx_one
        # above is more efficient than below:
        #  idx_one, idx_two = min(idx_one, idx_two), max(idx_one, idx_two)

        if not (idx_one, idx_two) in self._KM:
            self._KM[(idx_one, idx_two)] = \
                self.kernel(self._sample[idx_one, :],     # from 1st sample
                            self._sample_two[idx_two, :]) # from 2nd sample
                            # second refers to the first in the default case!
            self._num_ker_eval += 1

        return self._KM[(idx_one, idx_two)]


    def _features(self, index):
        """
        Returns the sample [features] corresponding to a given index.

        Using this would help abstract out the underlying data structure for samples and
            their features. For example, inputs can be simply CSVs, or numpy arrays or
            MLDataset or xarray or pandas etc.
        Disadvantages include the 2 extra function calls to be made for each kernel eval,
            which could be saved when operating on a predetermined format.
        """

        return self._sample[index, :]


    def __getitem__(self, index_obj):
        """
        Item getter to allow for efficient access
        to partial or random portions of kernel matrix!

        Indexing here is aimed to be compliant with numpy implementation
        as much as possible: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#arrays-indexing

        """

        if not len(index_obj) == 2 or not isinstance(index_obj, tuple):
            raise KMAccessError('Invalid attempt to access the kernel matrix '
                                '-: must supply two [sets/ranges of] indices in a tuple!')

        set_one, are_all_selected_dim_one = self._get_indices_in_sample(index_obj[0],
                                                                        dim=0)
        set_two, are_all_selected_dim_two = self._get_indices_in_sample(index_obj[1],
                                                                        dim=1)

        # below code prevents user from [VERY] inefficiently computing
        # the entire kernel matrix with KM[:,:],
        # without exploiting the fact that KM is symmetric
        if are_all_selected_dim_one and are_all_selected_dim_two:
            return self._populate_fully(fill_lower_tri=True)
        else:
            return self._compute_for_index_combinations(set_one, set_two)


    def _get_indices_in_sample(self, index_obj_per_dim, dim):
        """
        Turn an index or slice object on a given dimension
        into a set of row indices into sample the kernel matrix is attached to.

        As the kernel matrix is 2D and symmetric of known size,
            dimension size doesn't need to be specified, it is taken from self.num_samples

        """

        are_all_selected = False

        if isinstance(index_obj_per_dim, int) or np.isscalar(index_obj_per_dim):
            indices = [index_obj_per_dim, ]  # making it iterable
        elif isinstance(index_obj_per_dim, slice):
            if index_obj_per_dim is None:
                are_all_selected = True
            _slice_index_list = index_obj_per_dim.indices(self.shape[dim])
            indices = list(range(*_slice_index_list))  # *list expands it as args
        elif isinstance(index_obj_per_dim, Iterable) and \
            not isinstance(index_obj_per_dim, str):
            # TODO no restriction on float: float indices will be rounded down towards 0
            indices = list(map(int, index_obj_per_dim))
        else:
            raise KMAccessError('Invalid index method/indices for kernel matrix '
                                'of shape : {km_shape}.'
                                ' Only int, slice or iterable objects are allowed!'
                                ''.format(km_shape=self.shape))

        # enforcing constraints
        if any([index >= self.shape[dim] or index < 0 for index in indices]):
            raise KMAccessError('Invalid index method/indices for kernel matrix!\n'
                                ' Some indices in {} are out of range: '
                                ' shape : {km_shape},'
                                ' index values must all be >=0 and < corr. dimension'
                                ''.format(indices, km_shape=self.shape))

        # slice object returns empty list if all specified are out of range
        if len(indices) == 0:
            raise KMAccessError('No samples were selected in dim {}'.format(dim))

        # removing duplicates and sorting
        indices = sorted(list(set(indices)))

        if len(indices) == self.shape[dim]:
            are_all_selected = True

        return indices, are_all_selected


    def _compute_for_index_combinations(self, set_one, set_two):
        """Computes value of kernel matrix for all combinations of given set of indices"""

        return np.array([self._eval_kernel(idx_one, idx_two)
                         for idx_one, idx_two in iter_product(set_one, set_two)],
                        dtype=self._sample.dtype).reshape(len(set_one), len(set_two))


    def _populate_fully(self, dense_fmt=False, fill_lower_tri=False):
        """Applies the kernel function on all pairs of points in a sample.

        CAUTION: this may not always be necessary,
            and can take HUGE memory for LARGE datasets,
            and also can take a lot of time.

        """

        # kernel matrix is symmetric (in a single sample case)
        #   so we need only to STORE half the matrix!
        # as we are computing the full matrix anyways, it's better to keep a copy
        #   to avoid recomputing it for each access of self.full* attributes
        if not self._populated_fully and not hasattr(self, '_full_km'):
            if not dense_fmt:
                self._full_km = lil_matrix(self.shape, dtype=cfg.km_dtype)
            else:
                self._full_km = np.empty(self.shape, dtype=cfg.km_dtype)

            try:
                # kernel matrix is symmetric (in a single sample case)
                #   so we need only compute half the matrix!
                # computing the kernel for diagonal elements i,i as well
                #   as ix_two, even when equal to ix_one, refers to sample_two in the two_samples case
                for ix_one in range(self.shape[0]): # number of rows!
                    for ix_two in range(ix_one, self.shape[1]): # from second sample!
                        self._full_km[ix_one, ix_two] = self._eval_kernel(ix_one, ix_two)
            except:
                raise RuntimeError('Unable to fully compute the kernel matrix!')
            else:
                self._populated_fully = True

        if fill_lower_tri and not self._lower_tri_km_filled:
            try:
                # choosing k=-1 as main diag is already covered above (nested for loop)
                ix_lower_tri = np.tril_indices(self.shape[0], m=self.shape[1], k=-1)

                if not self._two_samples and self.shape[0] == self.shape[1]:
                    self._full_km[ix_lower_tri] = self._full_km.T[ix_lower_tri]
                else:
                    # evaluating it for the lower triangle as well!
                    for ix_one, ix_two in zip(*ix_lower_tri):
                        self._full_km[ix_one, ix_two] = self._eval_kernel(ix_one, ix_two)
            except:
                raise RuntimeError('Unable to symmetrize the kernel matrix!')
            else:
                self._lower_tri_km_filled = True

        if issparse(self._full_km) and dense_fmt:
            self._full_km = self._full_km.todense()

        return self._full_km


    def __str__(self):
        """human readable presentation"""

        string = "{}: {}".format(self.name, str(self.kernel))
        if self._sample is not None:
            # showing normalization status only when attached to data!
            string += " (normed={}) on {}".format(self._keep_normed, self._sample_descr)

        return string


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


    # TODO implement arithmetic operations on kernel matrices
    def __add__(self, other):
        """Addition"""
        raise NotImplementedError()


    def __mul__(self, other):
        """Multiplication"""
        raise NotImplementedError()


    def __sub__(self, other):
        """Subtraction"""
        raise NotImplementedError()


class KernelMatrixPrecomputed(object):
    """Convenience decorator for kernel matrices in ndarray or simple matrix format."""


    def __init__(self, matrix, name=None):
        """Constructor"""

        try:
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
        except:
            raise ValueError('Input matrix is not convertible to numpy array!')

        if matrix.ndim != 2 or not_symmetric(matrix):
            raise ValueError('Input matrix appears to be NOT 2D or symmetric! '
                             'Symmetry is needed for a valid kernel.')

        self._KM = matrix
        self.num_samples = self._KM.shape[0]

        if name is None:
            self.name = 'Precomputed'
        else:
            self.name = str(name)


    def __len__(self):
        """size of kernel matrix"""

        return self.size


    @property
    def size(self):
        """size of kernel matrix"""

        return self._KM.shape[0]


    @property
    def full(self):
        """Returns the full kernel matrix (in dense format, as its already precomputed)"""
        return self._KM


    @property
    def diag(self):
        """Returns the diagonal of the kernel matrix"""

        return self._KM.diagonal()


    def __getitem__(self, index_obj):
        """Access the matrix"""

        try:
            return self._KM[index_obj]
        except:
            raise KMAccessError('Invalid attempt to access the 2D kernel matrix!')


    def __str__(self):
        """human readable presentation"""

        return "{}(num_samples={})".format(self.name, self.num_samples)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class ConstantKernelMatrix(object):
    """Custom KM to represent a constant.

    TODO this is not fully though-out, or well. Consider removing or rewriting!
    """


    def __init__(self, num_samples=None, value=0.0, name=None, dtype='float'):

        self.num_samples = num_samples
        self.const_value = value
        self.dtype = dtype

        if name is None:
            self.name = 'Constant'
        else:
            self.name = str(name)


    def __len__(self):
        """size of kernel matrix"""

        return self.size


    @property
    def size(self):
        """size of kernel matrix"""
        return self.num_samples


    @property
    def full(self):
        """Returns the full kernel matrix (in dense format)"""

        if not hasattr(self, '_KM'):
            self._KM = np.full((self.num_samples, self.num_samples),
                               fill_value=self.const_value, dtype=self.dtype)

        return self._KM


    @property
    def diag(self):
        """Returns the diagonal of the kernel matrix"""

        return np.full((self.num_samples,),
                       fill_value=self.const_value, dtype=self.dtype)


    def __getitem__(self, index_obj):
        """Access the matrix"""

        try:
            return self._KM[index_obj]
        except:
            raise KMAccessError('Invalid attempt to access the 2D kernel matrix!')


    def __str__(self):
        """human readable presentation"""

        return "{}(value={},num_samples={})".format(self.name, self.const_value,
                                                    self.num_samples)


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


VALID_KERNEL_MATRIX_TYPES = (KernelMatrix, KernelMatrixPrecomputed, np.ndarray)


class KernelSet(object):
    """
    Container class to manage a set of compatible KernelMatrix instances.

    Compatibility is checked based on the size (number of samples they operate on).
    Provides methods to iterate over the KMs, access a subset and query the underlying kernel funcs.

    """


    def __init__(self,
                 km_set=None,
                 name='KernelSet',
                 num_samples=None):
        """Constructor."""

        self.name = name

        # empty to start with
        self._km_set = list()

        # user can choose to set the properties of the kernel matrices
        # this num_samples property is key, as only KMs with same value are allowed in
        if num_samples is not None:
            self._num_samples = num_samples
            self._is_init = True
        else:
            # to denote no KM has been added yet, or their size property is not set
            self._is_init = False
            self._num_samples = None

        if isinstance(km_set, Iterable):
            for km in km_set:
                self.append(km)
        elif isinstance(km_set, VALID_KERNEL_MATRIX_TYPES):
            self.append(km_set)
        elif km_set is None:
            pass  # do nothing
        else:
            raise TypeError('Unknown type of input matrix! '
                            'Must be one of:\n'
                            '{}'.format(VALID_KERNEL_MATRIX_TYPES))


    @property
    def size(self):
        """Number of kernel matrices in this set"""

        return len(self._km_set)


    @property
    def num_samples(self):
        """Number of samples in each individual kernel matrix """

        return self._num_samples


    def __len__(self):
        """Returns the number of kernels in this set"""

        return len(self._km_set)


    def append(self, KM):
        """
        Method to add a new kernel to the set.

        Checks to ensure the new KM is compatible in size to the existing set.
        """

        if not isinstance(KM, (BaseKernelFunction, KernelMatrix,
                               KernelMatrixPrecomputed)):
            KM = KernelMatrixPrecomputed(KM)

        if not self._is_init and self._num_samples is None:
            self._num_samples = copy(KM.num_samples)
            self._is_init = True

        if self._num_samples != KM.num_samples:
            raise KMSetAdditionError('Dimension of this KM {} is incompatible '
                                     'with KMSet of {}! '
                                     ''.format(KM.num_samples, self.num_samples))

        self._km_set.append(KM)


    def __getitem__(self, index):
        """To retrieve individual kernels"""

        if not isinstance(index, int):
            raise ValueError('Only integer indices are permitted, '
                             'accessing one KM at a time')

        if index < 0 or index >= self.size:
            raise IndexError('Index out of range for KernelSet of size {}'
                             ''.format(self.size))

        # TODO elements need to accessible by more than a simple integer index!
        #   Perhaps KernelMatrix can provide a hash to uniquely refer to an instance
        return self._km_set[index]


    def take(self, indices, name='SelectedKMs'):
        """Returns a new KernelSet with requested kernel matrices, identified by their indices."""

        indices = self._check_indices(indices)

        new_set = KernelSet(name=name)
        for idx in indices:
            # TODO should we add a copy of ith KM, or just a reference?
            #   No copy-->accidental changes!
            new_set.append(self._km_set[idx])

        return new_set


    def get_kernel_funcs(self, indices, name='SelectedKernelFuncs'):
        """
        Returns kernel functions underlying the specified kernel matrices in this kernel set.

        This is helpful to apply a given set of kernel functions on new sets of data (e.g. test set)

        """

        indices = self._check_indices(indices)

        return (self._km_set[idx].kernel for index in indices)


    def _check_indices(self, indices):
        """Checks the validity and type of indices."""

        if not isinstance(indices, Iterable):
            indices = [indices, ]

        indices = np.array(indices, dtype='int64')

        if any(indices < 0) or any(indices >= self.size):
            raise IndexError('One/more indices are out of range for KernelSet of size {}'
                             ''.format(self.size))

        return indices


    def __str__(self):
        """Human readable repr"""

        return "{}({} kernels, {} samples):\n\t{} " \
               "".format(self.name, self.size, self.num_samples,
                         "\n\t".join(map(str, self._km_set)))


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


    def __iter__(self):
        """Making an iterable."""

        for index in range(self.size):
            yield self._km_set[index]


    def attach_to(self, sample,
                  name='sample',
                  attr_name=None,
                  attr_value=None):
        """
        Attach all the kernel matrices in this set to a given sample.

        Any previous evaluations to other samples and their results will be reset.

        Parameters
        ----------
        sample : ndarray
            Input sample to operate on
            Must be 2D of shape (num_samples, num_features)

        name : str
            Identifier for the sample (esp. when multiple are in the same set)

        """

        self.sample = ensure_ndarray_2D(sample)
        if self._num_samples is not None and sample.shape[0] != self._num_samples:
            raise ValueError('Number of samples in input differ from this KernelSet')
        else:
            self._num_samples = sample.shape[0]

        for index in range(self.size):
            self._km_set[index].attach_to(sample, name_one=name)
            if attr_name is not None:
                self._km_set[index].set_attr(attr_name, attr_value)


    def extend(self, another_km_set):
        """Extends the current set by adding in all elements from another set."""

        if not isinstance(another_km_set, KernelSet):
            raise KMSetAdditionError('Input is not a KernelSet!'
                                     'Build a KernelSet() first.')

        if another_km_set.num_samples != self.num_samples:
            raise ValueError('The two KernelSets are not compatible (in size: # samples)')

        for km in another_km_set:
            self.append(km)


    def set_attr(self, name, values):
        """
        Sets user-defined attributes for the kernel matrices in this set.

        If len(values)==1, same value is set for all. Otherwise values must be of size as
        KernelSet, providing a separate value for each element.

        Useful to identify this kernel matrix in various aspects!
        You could think of them as tags or identifiers etc.
        As they are user-defined, they are ideal to represent user needs and applications.
        """

        if not isinstance(values, Iterable) or isinstance(values, str):
            values = [values]*self.size
        elif len(values) != self.size:
            raise ValueError('Values must be single element, or '
                             'of the same size as this KernelSet ({}), '
                             'providing a separate value for each element.'
                             'It is {}'.format(self.size, len(values)))

        for index in range(self.size):
            self._km_set[index].set_attr(name, values[index])


    def get_attr(self, name, value_if_not_found=None):
        """Returns the value of an user-defined attribute.

        If not set previously, or no match found, returns value_if_not_found.
        """

        return [ self._km_set[index].get_attr(name, value_if_not_found)
                 for index in range(self.size) ]


class CompositeKernel(ABC):
    """Class to combine a set of kernels into a composite kernel."""


    def __init__(self, km_set, name='Composite'):
        """Constructor."""

        if not isinstance(km_set, KernelSet):
            raise TypeError('Input must be a KernelSet')

        if km_set.size < 2:
            raise ValueError('KernelSet must have atleast 2 kernels')

        if km_set.num_samples is None:
            raise ValueError('KernelSet is not attached to any sample!')

        self.km_set = km_set
        self.num_samples = km_set.num_samples
        self._is_fitted = False
        self.name = name


    @abstractmethod
    def fit(self):
        """Abstract methods that needs to be defined later."""
        pass


    @property
    def composite_KM(self):
        """Returns the result of composite operation"""

        if self._is_fitted:
            return self.KM
        else:
            raise ValueError('{} is not fitted yet!'.format(self.name))


    def __str__(self):
        """human readable presentation"""

        return "{}-->{}".format(self.name, str(self.km_set))


    # aliasing them to __str__ for now
    __format__ = __str__
    __repr__ = __str__


class SumKernel(CompositeKernel):
    """Class to define and compute a weighted sum kernel from a KernelSet"""


    def __init__(self, km_set, name='SumKernel'):
        """Constructor."""

        super().__init__(km_set, name=name)


    def fit(self, kernel_weights=None):
        """Computes the sum kernel"""

        if kernel_weights is None:
            kernel_weights = np.ones(self.km_set.size)
        else:
            kernel_weights = ensure_ndarray_1D(kernel_weights)
            if kernel_weights.size != self.km_set.size:
                raise ValueError('Incompatible set of kernel_weights given.'
                                 'Must be an array of length exactly {}'
                                 ''.format(self.km_set.size))

        self.KM = np.zeros((self.num_samples, self.num_samples))
        for weight, km in zip(kernel_weights, self.km_set):
            self.KM = self.KM + weight * km.full

        self._is_fitted = True


class ProductKernel(CompositeKernel):
    """Class to define and compute a Product kernel from a KernelSet"""


    def __init__(self, km_set, name='ProductKernel'):
        """Constructor."""

        super().__init__(km_set, name=name)


    def fit(self):
        """Computes the product kernel."""

        self.KM = np.ones((self.num_samples, self.num_samples))
        for km in self.km_set:
            self.KM = self.KM * km.full  # * is element-wise multiplication here

        self._is_fitted = True


class AverageKernel(CompositeKernel):
    """Class to define and compute an Average kernel from a KernelSet"""


    def __init__(self, km_set, name='AverageKernel'):
        """Constructor."""

        super().__init__(km_set, name=name)


    def fit(self):
        """Computes the average kernel"""

        self.KM = np.zeros((self.num_samples, self.num_samples))
        for km in self.km_set:
            self.KM = self.KM + km.full  # * is element-wise multiplication here

        # dividing by N, to make it an average
        self.KM = self.KM / self.km_set.size

        self._is_fitted = True


class WeightedAverageKernel(CompositeKernel):
    """Class to define and compute a weighted verage kernel from a KernelSet"""


    def __init__(self,
                 km_set,
                 weights,
                 name='WeightedAverageKernel'):
        """Constructor."""

        super().__init__(km_set, name=name)

        if self.km_set.size == len(weights):
            self.weights = ensure_ndarray_1D(weights)
        else:
            raise ValueError(
                'Number of weights ({}) supplied differ from the kernel set size ({})'
                ''.format(self.km_set.size, len(weights)))


    def fit(self):
        """Computes the weighted average kernel"""

        self.KM = np.zeros((self.num_samples, self.num_samples))
        for weight, km in zip(self.weights, self.km_set):
            self.KM = self.KM + weight * km.full

        self._is_fitted = True


class PredictiveModelFromKernelMatrix(KernelMatrix):
    """Class to turn a given kernel into a predictive model"""

    pass  # raise NotImplementedError()
