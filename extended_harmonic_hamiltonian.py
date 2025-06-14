@typeguard.typechecked
@dataclass
class ExtendedHarmonicFunction(EnergyFunction):
    positions: np.ndarray
    cell: np.ndarray
    extended_hessian: np.ndarray
    energy: Optional[float] = None
    hessian_coordinates: str = "sh"
    _Ts: np.ndarray | None = None
    _Tc: np.ndarray | None = None
    _transformed_pos_opt: np.ndarray | None = None

    def __post_init__(self):
        # defintion _Ts:  s . _Ts = hessian_coords[0]
        if self.hessian_coordinates[0] == "s":
            self._Ts = np.eye(3)  # s = s
        elif self.hessian_coordinates[0] == "d":
            self._Ts = self.cell  # s . h_0 = d
        else:
            raise NotImplementedError(
                f"Coordinate transformation {self.hessian_coordinates} not implemented"
            )
        # defintion _Tc: _Tc . cell = hessian_coords[1]
        if self.hessian_coordinates[1] == "h":
            self._Tc = np.eye(3)
        elif self.hessian_coordinates[1] == "f":
            self._Tc = np.linalg.inv(self.cell)
        else:
            raise NotImplementedError(
                f"Coordinate transformation {self.hessian_coordinates} not implemented"
            )
        # optimized positions in scaled or deformed space
        self._transformed_pos_opt = self.positions @ np.linalg.inv(self.cell) @ self._Ts
        

    def __call__(
        self,
        geometry: Geometry,
    ) -> dict[str, float | np.ndarray]:
        """ 
        We assume that the positions naturally scale with the cell, leading to (ds/dh)=0 and (dr_i/dh_ji)=s_j
        This is a normal assumption, and this is probably also assumed when calculating stresses or applying barostats
        """

        # definition Tr:  r . Tr = s
        Tr = np.linalg.inv(geometry.cell)
        transformed_pos_sample = geometry.per_atom.positions @ Tr @ self._Ts

        diff_pos = transformed_pos_sample - self._transformed_pos_opt
        diff_cell = self._Tc @ (geometry.cell - self.cell)
        delta = np.concatenate((diff_pos.reshape(-1), diff_cell.reshape(-1) ))
        grad_T = np.dot(self.extended_hessian, delta)
        energy = 0.5 * np.dot(delta, grad_T)
        if self.energy is not None:
            energy += self.energy

        # from grad_T to forces (with chain rule!)
        Trs = Tr @ self._Ts
        grad_pos = grad_T[:-9].reshape(-1, 3) @ Trs.T
        forces = (-1.0) * grad_pos

        # from cell derivative to stresses
        grad_cell = self._Tc.T @ grad_T[-9:].reshape(3, 3)
        volume = np.linalg.det(geometry.cell)
        stress = (1/volume) * grad_cell.T @ geometry.cell

        stress = np.copy((stress+stress.T)/2)
        return {"energy": energy, "forces": forces, "stress": stress}
