"""
Inspired from https://github.com/google/svcca, with some modifications and bug fixes
"""
import copy
import numpy as np


class CCA:
    def __init__(self, acts1, acts2):
        """
        Args:
                acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                        datapoints where entry (i,j) is the output of neuron i on
                        datapoint j.
                acts2: (num_neurons2, data_points) same as above, but (potentially)
                        for a different set of neurons. Note that acts1 and acts2
                        can have different numbers of neurons, but must agree on the
                        number of datapoints
        """
        # assert dimensionality equal
        assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"

        self.means1 = np.mean(acts1, axis=1, keepdims=True)
        self.means2 = np.mean(acts2, axis=1, keepdims=True)
        self.view1 = acts1
        self.view2 = acts2

    def get_cca_parameters(self, epsilon_x, epsilon_y, verbose=True):
        """
        The main function for computing cca similarities.
        This function computes the cca similarity between two sets of activations,
        returning a dict with the cca coefficients, a few statistics of the cca
        coefficients, and (optionally) the actual directions.
        Args:
                epsilon_x: small float to help stabilize computations (sp. to view1)
                epsilon_y: small float to help stabilize computations (sp. to view2)
                verbose: Boolean, whether intermediate outputs are printed
        Returns:
                return_dict: A dictionary with outputs from the cca computations.
                            Contains neuron coefficients (combinations of neurons
                            that correspond to cca directions), the cca correlation
                            coefficients (how well aligned directions correlate),
                            x and y idxs (for computing cca directions on the fly
                            if compute_dirns=False), and summary statistics. If
                            compute_dirns=True, the cca directions are also
                            computed.
        """
        return_dict = {}

        # compute covariance with numpy function for extra stability
        numx = self.view1.shape[0]
        numy = self.view2.shape[0]

        covariance = np.cov(self.view1, self.view2)
        sigma_xx = covariance[:numx, :numx]
        sigma_xy = covariance[:numx, numx:]
        sigma_yx = covariance[numx:, :numx]
        sigma_yy = covariance[numx:, numx:]

        # rescale covariance to make cca computation more stable
        xmax = np.max(np.abs(sigma_xx))
        ymax = np.max(np.abs(sigma_yy))
        sigma_xx /= xmax
        sigma_yy /= ymax
        sigma_xy /= np.sqrt(xmax * ymax)
        sigma_yx /= np.sqrt(xmax * ymax)

        (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = self.remove_small(
            epsilon_x, epsilon_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy
        )

        sigma_xx += epsilon_x * np.eye(sigma_xx.shape[0])
        sigma_yy += epsilon_y * np.eye(sigma_yy.shape[0])
        ([u, s, v], invsqrt_xx, invsqrt_yy) = self.train_cca(
            sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs, verbose=verbose
        )
        # if x_idxs or y_idxs is all false, return_dict has zero entries
        if (not np.any(x_idxs)) or (not np.any(y_idxs)):
            return self.create_zero_dict(compute_dirns, self.view1.shape[1])

        x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
        y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

        return_dict["coef_x"] = u.T
        return_dict["invsqrt_xx"] = invsqrt_xx
        return_dict["full_coef_x"] = np.zeros((numx, numx))
        np.place(return_dict["full_coef_x"], x_mask, return_dict["coef_x"])
        return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
        np.place(return_dict["full_invsqrt_xx"], x_mask, return_dict["invsqrt_xx"])

        return_dict["coef_y"] = v
        return_dict["invsqrt_yy"] = invsqrt_yy
        return_dict["full_coef_y"] = np.zeros((numy, numy))
        np.place(return_dict["full_coef_y"], y_mask, return_dict["coef_y"])
        return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
        np.place(return_dict["full_invsqrt_yy"], y_mask, return_dict["invsqrt_yy"])

        proj_mat_x = np.dot(return_dict["coef_x"], return_dict["invsqrt_xx"])
        proj_mat_y = np.dot(return_dict["coef_y"], return_dict["invsqrt_yy"])

        return_dict["cca_coef1"] = s
        return_dict["cca_coef2"] = s
        return_dict["x_idxs"] = x_idxs
        return_dict["y_idxs"] = y_idxs
        return_dict["proj_mat_x"] = proj_mat_x
        return_dict["proj_mat_y"] = proj_mat_y

        return return_dict

    def create_zero_dict(self, compute_dirns, dimension):
        """Outputs a zero dict when neuron activation norms too small.
        This function creates a return_dict with appropriately shaped zero entries
        when all neuron activations are very small.
        Args:
                    compute_dirns: boolean, whether to have zero vectors for directions
                    dimension: int, defines shape of directions
        Returns:
                    return_dict: a dict of appropriately shaped zero entries
        """
        return_dict = {}
        return_dict["cca_coef1"] = np.asarray(0)
        return_dict["cca_coef2"] = np.asarray(0)
        return_dict["idx1"] = 0
        return_dict["idx2"] = 0

        if compute_dirns:
            return_dict["cca_dirns1"] = np.zeros((1, dimension))
            return_dict["cca_dirns2"] = np.zeros((1, dimension))

        return return_dict

    def positivedef_matrix_sqrt(self, array):
        """Stable method for computing matrix square roots, supports complex matrices.
        Args:
                    array: A numpy 2d array, can be complex valued that is a positive
                        definite symmetric (or hermitian) matrix
        Returns:
                    sqrtarray: The matrix square root of array
        """
        w, v = np.linalg.eigh(array)
        #  A - np.dot(v, np.dot(np.diag(w), v.T))
        wsqrt = np.sqrt(w)
        sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
        return sqrtarray

    def remove_small(
        self, epsilon_x, epsilon_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy
    ):
        """Takes covariance between X, Y, and removes values of small magnitude.
        Args:
                    epsilon_x : cutoff value for norm below which directions are thrown
                            away
                    epsilon_y : cutoff value for norm below which directions are thrown
                            away
                    sigma_xx: 2d numpy array, variance matrix for x
                    sigma_xy: 2d numpy array, crossvariance matrix for x,y
                    sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                            (conjugate) transpose of sigma_xy
                    sigma_yy: 2d numpy array, variance matrix for y
        Returns:
                    sigma_xx_crop: 2d array with low x norm directions removed
                    sigma_xy_crop: 2d array with low x and y norm directions removed
                    sigma_yx_crop: 2d array with low x and y norm directiosn removed
                    sigma_yy_crop: 2d array with low y norm directions removed
                    x_idxs: indexes of sigma_xx that were removed
                    y_idxs: indexes of sigma_yy that were removed
        """

        x_diag = np.abs(np.diagonal(sigma_xx))
        y_diag = np.abs(np.diagonal(sigma_yy))
        x_idxs = x_diag >= epsilon_x
        y_idxs = y_diag >= epsilon_y

        sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
        sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
        sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
        sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

        return (
            sigma_xx_crop,
            sigma_xy_crop,
            sigma_yx_crop,
            sigma_yy_crop,
            x_idxs,
            y_idxs,
        )

    def train_cca(
        self, sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs, verbose=True
    ):
        """Main cca computation function, takes in variances and crossvariances.
        This function takes in the covariances and cross covariances of X, Y,
        preprocesses them (removing small magnitudes) and outputs the raw results of
        the cca computation, including cca directions in a rotated space, and the
        cca correlation coefficient values.
        Args:
                    sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                            variance matrix for x
                    sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                            crossvariance matrix for x,y
                    sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                            crossvariance matrix for x,y (conj) transpose of sigma_xy
                    sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                            variance matrix for y
                    x_idxs: The indexes of the input sigma_xx that were pruned
                            by remove_small
                    y_idxs:       Same as above but for sigma_yy
                    verbose:  boolean on whether to print intermediate outputs
        Returns:
                    [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                                ux and vx are (conj) transposes of each other, being
                                the canonical directions in the X subspace.
                                sx is the set of canonical correlation coefficients-
                                how well corresponding directions in vx, Vy correlate
                                with each other.
                    [uy, sy, vy]: Same as above, but for Y space
                    invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                                directions back to original space
                    invsqrt_yy:   Same as above but for sigma_yy

        """
        # check that acts1, acts2 are transposition
        assert self.view1.shape[0] < self.view1.shape[1], (
            "input must be number of neurons" "by datapoints"
        )
        assert self.view2.shape[0] < self.view2.shape[1], (
            "input must be number of neurons" "by datapoints"
        )
        numx = sigma_xx.shape[0]
        numy = sigma_yy.shape[0]

        if numx == 0 or numy == 0:
            return (
                [0, 0, 0],
                [0, 0, 0],
                np.zeros_like(sigma_xx),
                np.zeros_like(sigma_yy),
                x_idxs,
                y_idxs,
            )

        if verbose:
            print("adding eps to diagonal and taking inverse")

        inv_xx = np.linalg.pinv(sigma_xx)
        inv_yy = np.linalg.pinv(sigma_yy)
        if verbose:
            print("taking square root")
        invsqrt_xx = self.positivedef_matrix_sqrt(inv_xx)
        invsqrt_yy = self.positivedef_matrix_sqrt(inv_yy)
        if verbose:
            print("dot products...")
        arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

        if verbose:
            print("trying to take final svd")
        u, s, v = np.linalg.svd(arr)

        if verbose:
            print("computed everything!")

        return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy

    def get_cca_coefficients(
        self, proj_mat_x, proj_mat_y, x_idxs, y_idxs, num_directions
    ):
        """Computes projection weighting for weighting CCA coefficients

        Args:
            proj_mat_x: square projection matrix of size valid indices in x_idxs <= d1
            proj_mat_y: square projection matrix of size valid indices in y_idxs <= d2
            x_idxs: boolean array for view 1 indices corresponding to "valid" dimensions;
                    size (d1,)
            y_idxs: boolean array for view 2 indices corresponding to "valid" dimensions;
                    size (d2,)
            num_directions: min(#valid indices in view1, #valid indices in view2)

        Returns:
            Optimum CCA correlation values for the two views
            Projection of view 1; size num_directions x N
            Projection of view 2; size num_directions x N
        """
        proj_1 = np.dot(
            proj_mat_x[:num_directions], (self.view1[x_idxs] - self.means1[x_idxs])
        )
        proj_2 = np.dot(
            proj_mat_y[:num_directions], (self.view2[y_idxs] - self.means2[y_idxs])
        )
        corr_scores = np.einsum("ij,ij->i", proj_1, proj_2)
        vec_norms = np.sqrt(
            np.multiply(
                np.einsum("ij,ij->i", proj_1, proj_1),
                np.einsum("ij,ij->i", proj_2, proj_2),
            )
        )
        corr_scores /= vec_norms

        return corr_scores, proj_1, proj_2

    def compute_weighted_sum(self, acts, acts_means, dirns, coefs):
        """Computes weights for projection weighing"""
        dirns += acts_means
        P, _ = np.linalg.qr(dirns.T)
        weights = np.sum(np.abs(np.dot(P.T, acts.T)), axis=1)
        weights = weights / np.sum(weights)
        return np.sum(weights * coefs)

    def compute_pwcca(self, proj_mat_x, proj_mat_y, x_idxs, y_idxs, mean_score=False):
        """Computes projection weighting for weighting CCA coefficients

        Args:
            proj_mat_x: square projection matrix of size valid indices in x_idxs <= d1
            proj_mat_y: square projection matrix of size valid indices in y_idxs <= d2
            x_idxs: boolean array for view 1 indices corresponding to "valid" dimensions;
                    size (d1,)
            y_idxs: boolean array for view 2 indices corresponding to "valid" dimensions;
                    size (d2,)

        Returns:
            Projection weighted mean of cca coefficients
        """
        num_directions = np.min([len(proj_mat_x), len(proj_mat_y)])

        corr_scores, view1_projected, view2_projected = self.get_cca_coefficients(
            proj_mat_x, proj_mat_y, x_idxs, y_idxs, num_directions
        )

        score_x = self.compute_weighted_sum(
            self.view1[x_idxs][:num_directions],
            self.means1[x_idxs][:num_directions],
            view1_projected,
            corr_scores,
        )
        score_y = self.compute_weighted_sum(
            self.view2[y_idxs][:num_directions],
            self.means2[y_idxs][:num_directions],
            view2_projected,
            corr_scores,
        )
        if mean_score:  # not a part of the original impementation
            return (score_x + score_y) / 2
        elif len(proj_mat_x) < len(proj_mat_y):
            return score_x
        else:
            return score_y

    def get_cca_score(
        self,
        train=True,
        epsilon_x=None,
        epsilon_y=None,
        proj_mat_x=None,
        proj_mat_y=None,
        x_idxs=None,
        y_idxs=None,
        mean_score=False,
    ):
        """
        Returns CCA score
            in train mode if train=True or
            in eval mode if train=False and the parameters are passed to the function
        """
        if train:
            assert epsilon_x is not None, "Missing view 1 reg parameter"
            assert epsilon_y is not None, "Missing view 2 reg parameter"
            sresults = self.get_cca_parameters(epsilon_x, epsilon_y, verbose=False)
            proj_mat_x = sresults["proj_mat_x"]
            proj_mat_y = sresults["proj_mat_y"]
            x_idxs = sresults["x_idxs"]
            y_idxs = sresults["y_idxs"]
        else:
            assert proj_mat_x is not None, "Missing view 1 projection"
            assert x_idxs is not None, "Missing view 1 indices"
            assert proj_mat_y is not None, "Missing view 2 projection"
            assert y_idxs is not None, "Missing view 2 indices"

        similarity_score = self.compute_pwcca(
            proj_mat_x, proj_mat_y, x_idxs, y_idxs, mean_score
        )

        return similarity_score, (proj_mat_x, proj_mat_y, x_idxs, y_idxs)
