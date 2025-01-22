import numpy as np
from scipy import linalg

from predpca.utils import pcacov


class PredPCA:
    def __init__(
        self,
        kp_list: list[int] | range,
        prior_s_: float,
        gain: np.ndarray | None = None,  # (Nu, Ns)
    ):
        self.kp_list = kp_list
        self.prior_s_ = prior_s_
        self.gain = gain
        self.Q = None

    def fit(
        self,
        s: np.ndarray,  # (Ns, n_seq, seq_len) | (Ns, T)
        s_target: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
    ):
        s, s_target = _check_inputs(s, s_target)
        s_ = self._create_basis_functions(s)  # (Ns*Kp, T)
        return self._fit(s_, s_target)

    def _fit(
        self,
        s_: np.ndarray,  # (Ns*Kp, T)
        s_target: np.ndarray,  # (Kf, Ns, T)
    ):
        self.Q = self._compute_Q(s_, s_target)  # (Kf, Ns, Ns*Kp)
        return self

    def transform(
        self,
        s: np.ndarray,  # (Ns, n_seq, seq_len) | (Ns, T)
    ) -> np.ndarray:
        s, _ = _check_inputs(s)
        s_ = self._create_basis_functions(s)  # (Ns*Kp, T)
        return self._transform(s_)  # (Kf, Ns, T) | (Ns, T)

    def _transform(
        self,
        s_: np.ndarray,  # (Ns*Kp, T)
    ) -> np.ndarray:
        se = self._predict_input(s_)  # (Kf, Ns, T)
        return _check_outputs(se)  # (Kf, Ns, T) | (Ns, T)

    def fit_transform(
        self,
        s: np.ndarray,  # (Ns, n_seq, seq_len) | (Ns, T)
        s_target: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
    ) -> np.ndarray:
        s, s_target = _check_inputs(s, s_target)
        s_ = self._create_basis_functions(s)  # (Ns*Kp, T)
        return self._fit(s_, s_target)._transform(s_)  # (Kf, Ns, T) | (Ns, T)

    def _create_basis_functions(
        self,
        s: np.ndarray,  # (Ns, n_seq, seq_len)
    ) -> np.ndarray:
        if self.gain is not None:
            s = np.einsum("ij, jkl -> ikl", self.gain, s)  # (Nu, n_seq, seq_len)

        enc_dim, n_seq, seq_len = s.shape
        Kp = len(self.kp_list)

        s_ = np.zeros((enc_dim * Kp, n_seq * seq_len))
        for i, k in enumerate(self.kp_list):
            s_[enc_dim * i : enc_dim * (i + 1)] = np.roll(s, k, axis=-1).reshape(enc_dim, -1)

        return s_

    def _compute_Q(
        self,
        s_: np.ndarray,  # (Ns*Kp, T)
        s_target: np.ndarray,  # (Kf, Ns, T)
    ) -> np.ndarray:
        Kf, Ns, _ = s_target.shape
        Nphi = s_.shape[0]

        S_S_inv = linalg.inv(s_ @ s_.T + np.eye(Nphi) * self.prior_s_)

        Q = np.empty((Kf, Ns, Nphi))
        for k in range(Kf):
            Q[k] = s_target[k] @ s_.T @ S_S_inv  # (Ns, N*Kp)

        return Q

    def _predict_input(
        self,
        s_: np.ndarray,  # (Ns*Kp, T)
    ) -> np.ndarray:
        Kf, Ns, _ = self.Q.shape
        T = s_.shape[1]

        se = np.empty((Kf, Ns, T))
        for k in range(Kf):
            se[k] = self.Q[k] @ s_  # (Ns, T)

        return se  # (Kf, Ns, T)

    def predict_input_pca(
        self,
        se: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
    ):
        se = _unsqueeze_kf(se)  # (Kf, Ns, T)
        Kf, Ns, T = se.shape

        # predict input covariance
        qSigmase = np.empty((Kf, Ns, Ns))
        for k in range(Kf):
            qSigmase[k] = se[k] @ se[k].T / T  # (Ns, Ns)

        qSigmase_mean = qSigmase.mean(axis=0)

        C, L, _ = pcacov(qSigmase_mean)  # C: (Ns, Ns), L: (Ns,)

        return C, L, qSigmase_mean

    def inverse_transform(
        self,
        se: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
    ) -> np.ndarray:  # (Ns, T)
        """Inverse transform from predicted states to basis functions and states

        Args:
            se: Predicted states
        Returns:
            s: Reconstructed states
        """
        se = _unsqueeze_kf(se)  # (Kf, Ns, T)
        Kf = se.shape[0]

        # Inverse Q transformation for each future step
        s_list = []
        for k in range(Kf):
            Q_pinv = linalg.pinv(self.Q[k])  # (Ns*Kp, Ns)
            s_ = Q_pinv @ se[k]  # (Ns*Kp, T)
            s = self._inverse_create_basis_functions(s_)  # (Ns, T)
            s_list.append(s)

        # Average over future steps
        s = np.mean(s_list, axis=0)  # (Ns, T)
        return s

    def _inverse_create_basis_functions(
        self,
        s_: np.ndarray,  # (Ns*Kp, T)
    ) -> np.ndarray:  # (Ns, T)
        """Inverse transform of basis functions

        Args:
            s_: Basis functions
        Returns:
            s: Reconstructed states
        """
        n_phi, T = s_.shape
        enc_dim = n_phi // len(self.kp_list)

        s_acc = np.zeros((enc_dim, T))
        for i, k in enumerate(self.kp_list):
            s_chunk = s_[enc_dim * i : enc_dim * (i + 1)]  # (enc_dim, T)
            s_rolled = np.roll(s_chunk, -k, axis=-1)
            s_acc += s_rolled
        s = s_acc / len(self.kp_list)

        if self.gain is not None:
            gain_pinv = linalg.pinv(self.gain)  # (Ns, Nu)
            s = gain_pinv @ s  # (Ns, T)

        return s


def _check_inputs(
    s: np.ndarray,  # (Ns, n_seq, seq_len) | (Ns, T)
    s_target: np.ndarray | None = None,  # (Kf, Ns, T) | (Ns, T)
) -> tuple[np.ndarray, np.ndarray]:
    s = _unsqueeze_seq(s)  # (Ns, n_seq, seq_len)
    if s_target is not None:
        s_target = _unsqueeze_kf(s_target)  # (Kf, Ns, T)
    return s, s_target


def _check_outputs(
    se: np.ndarray,  # (Kf, Ns, T) | (Ns, T)
) -> np.ndarray:
    return _squeeze_kf(se)  # (Kf, Ns, T) | (Ns, T)


def _unsqueeze_kf(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[np.newaxis]  # (1, Ns, T)
    return x


def _squeeze_kf(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        x = x[0]
    return x


def _unsqueeze_seq(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, np.newaxis, :]  # (Ns, n_seq, seq_len)
    return x
