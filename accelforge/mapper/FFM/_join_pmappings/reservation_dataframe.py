from collections import defaultdict

from numbers import Number
from typing import Callable, NamedTuple

from pandas import DataFrame, Series

from accelforge._accelerated_imports import pd, np

from accelforge.mapper.FFM._pareto_df.df_convention import *

npmax = np.maximum
npmin = np.minimum


class ReservationData(NamedTuple):
    key: ReservationKey
    df: pd.DataFrame

    @classmethod
    def from_args(
        cls, df: pd.DataFrame, resource: str, index: int
    ) -> "ReservationData":
        key = ReservationKey(resource, index, left=False)
        return ReservationData.from_df(df, key)

    @classmethod
    def from_df(cls, df: pd.DataFrame, key: ReservationKey) -> "ReservationData":
        return cls(key._replace(left=False), df)

    @property
    def right_col(self) -> str:
        return reservationkey2sizecol(self.key)

    @property
    def left_col(self) -> str:
        col = reservationkey2sizecol(self.key._replace(left=True))
        return col if col in self.df.columns else None

    @property
    def iters_above_col(self) -> str:
        return reservationkey2iterscol(self.key)

    @property
    def right(self) -> Series:
        return self.df[self.right_col]

    @property
    def left(self) -> Series:
        left_col = self.left_col
        return None if left_col is None else self.df[left_col]

    @property
    def iters_above(self) -> Series:
        return self.df[self.iters_above_col]

    @property
    def resource(self) -> str:
        return self.key.name

    @property
    def index(self) -> int:
        return self.key.index


def _find_reservation_size(
    datas: list[ReservationData], iters_above: Series, include_equal: bool = True
):
    result = 0
    compare = lambda x, y: x <= y if include_equal else x < y
    for r in datas:
        above = compare(r.iters_above, iters_above)
        result = npmax(result, np.where(above, r.right, 0))
    return result


def _reservation_size_above(datas: list[ReservationData], iters_above):
    return _find_reservation_size(datas, iters_above, include_equal=False)


def _reservation_size_at_or_above(datas: list[ReservationData], iters_above: Series):
    return _find_reservation_size(datas, iters_above, include_equal=True)


class ReservationDataFrame(DataFrame):
    """
    Reservations are each tracked with two columns, one for the reservation's size and
    one for the number of itxerations above the reservation (i.e., total number of times
    the reservation is replaced). This is used to tell how far down the LoopTree the
    reservation is (since lower in the LoopTree = more iterations above, larger shape).

    INVARIANTS:

    - Higher index -> higher # iterations above
    - Higher index -> same-or-larger reservation size (since it's inclusive)
    - One left+right reservation per index, sharing the same # iterations above

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_all_reservations()

    def assert_reservations_ordered(self):
        # NOTE: May be violated mid-merge
        for _, datas in self._get_all_reservations().items():
            assert tuple(d.index for d in datas) == tuple(range(len(datas)))
            for i in range(len(datas) - 1):
                # Iterations above is increasing
                leq = datas[i].iters_above <= datas[i + 1].iters_above
                assert leq.all(), f"Reservations out of order"

                # Tree is inclusive & the current right is a parent of below right.
                # NOTE: Left is not necessarily inclusive (e.g., if data was just
                # reserved on the right branch or a join just happened). It will be
                # inclusive after drop_redundant_data because we'll max right -> left
                leq = datas[i].right <= datas[i + 1].right
                assert leq.all(), f"Reservations out of order"

    def assert_data_not_redundant(self):
        for _, datas in self._get_all_reservations().items():
            for i in range(len(datas) - 1):
                prev = datas[i]
                r = datas[i + 1]

                # If number of iterations matches (i.e., same level), sizes should match
                eq = prev.iters_above == r.iters_above
                size_neq = prev.right != r.right
                assert not (eq & size_neq).any(), f"Reservations vary for a level"
                if prev.left is not None and r.left is not None:
                    left_ne = datas[i].left != datas[i + 1].left
                    assert not (eq & left_ne).any(), f"Reservations vary within a level"

                # If the left column of the lower level doesn't contribute anything
                # beyond that of the upper level, then it should have been removed
                if datas[i + 1].left is not None:
                    left_needed = datas[i + 1].left != datas[i + 1].right
                    left_needed &= ~eq  # Only care about the ones not in datas[i]
                    assert left_needed.any(), "Redundant left reservation"

                # If the right & left sizes of the lower level match that of the upper
                # level, number of iterations should match
                eq = datas[i].right == datas[i + 1].right
                if datas[i + 1].left is not None:
                    eq &= datas[i].left == datas[i + 1].left
                assert not eq.all(), "Redundant reservations"

    def set_reservation_column_numbers(self):
        renames = {}
        for resource, datas in self._get_all_reservations().items():
            for j, r in enumerate(datas):
                if r.index == j:
                    continue
                new = ReservationKey(resource, j, left=False)
                renames[r.right_col] = reservationkey2sizecol(new)
                if r.left_col is not None:
                    renames[r.left_col] = reservationkey2sizecol(new, left=True)
                renames[r.iters_above_col] = reservationkey2iterscol(new)
        if renames:
            self.rename(columns=renames, inplace=True)

    def _get_all_reservations(self) -> dict[str, list[ReservationData]]:
        """
        Returns {resource: ReservationData for each index in increasing order}
        """
        indices = defaultdict(set)
        cols_set = set(self.columns)
        for c in self.columns:
            if (key := col2reservationsize(c)) is not None:
                indices[key.name].add(key.index)
                a = reservationkey2iterscol(key)
                b = reservationkey2sizecol(key, left=False)
                assert all(col in cols_set for col in (a, b)), f"Missing columns"
        reservations = {}
        for r, idx in indices.items():
            cur_res = [ReservationData.from_args(self, r, i) for i in sorted(idx)]
            reservations[r] = cur_res
        return reservations

    def sort_reservations(self):
        """
        Puts puts reservation levels in increasing (iters_above, right size) order
        """
        self.set_reservation_column_numbers()
        
        for _, datas in self._get_all_reservations().items():
            if len(datas) < 2:
                continue
            iters = np.column_stack([r.iters_above for r in datas])
            rights = np.column_stack([r.right for r in datas])

            # This sorts by (iters_above, right)
            order = np.argsort(rights, axis=1, kind="stable")
            sorted_by_rights = np.argsort(
                np.take_along_axis(iters, order, axis=1), axis=1, kind="stable"
            )
            order = np.take_along_axis(order, sorted_by_rights, axis=1)

            if not (order != np.arange(order.shape[1])).any():
                continue

            lefts = np.column_stack(
                [r.right if r.left is None else r.left for r in datas]
            )
            iters = np.take_along_axis(iters, order, axis=1)
            rights = np.take_along_axis(rights, order, axis=1)
            lefts = np.take_along_axis(lefts, order, axis=1)

            for j, r in enumerate(datas):
                self.loc[:, r.right_col] = rights[:, j]
                self.loc[:, r.iters_above_col] = iters[:, j]
                left_col = reservationkey2sizecol(r.key, left=True)
                if (lefts[:, j] > rights[:, j]).any():
                    self.loc[:, left_col] = lefts[:, j]
                elif r.left_col is not None:
                    self.drop(columns=[left_col], inplace=True)

    def _drop_redundant_maybe_shift(self, datas: list[ReservationData]) -> bool:
        changed = np.zeros(len(self))
        for i in range(len(datas) - 1, -1, -1):

            def _update_changed(mask):
                nonlocal changed
                if mask.any():
                    changed = np.where(mask, npmax(changed, i), changed)

            r = datas[i]
            prev = datas[i - 1] if i > 0 else None
            prevprev = datas[i - 2] if i > 1 else None

            # If we're at the same level as the previous level, match up the
            # reservations.
            if i > 0:
                mask = r.iters_above == prev.iters_above
                if mask.any():
                    diff = mask & (r.right != prev.right)
                    if r.left is not None and prev.left is not None:
                        diff |= mask & (r.left != prev.left)
                    self._shift_reservation_up(r, prev, mask)
                    if r.left is not None and prev.left is not None:
                        self._masked_replace(prev.left_col, r.left_col, mask)
                    _update_changed(diff)

            # If we don't contribute anything over and above the previous reservation,
            # we're redundant & can be shifted upward.
            if i > 0:
                mask = r.right == prev.right
                if r.left_col is not None:
                    p_left = prev.right if prev.left_col is None else prev.left
                    mask &= r.left <= p_left
                mask &= r.iters_above != prev.iters_above
                if mask.any():
                    self._shift_reservation_up(prev, r, mask, iters_only=True)
                    _update_changed(mask)

            # if i > 0:
            #     same_level = r.iters_above == prev.iters_above
            #     mask = same_level & (r.right != prev.right)
            #     if mask.any():
            #         self._masked_replace(r.right_col, prev.right_col, mask)

            #     if r.left_col is not None:
            #         if prev.left_col is None:
            #             left_col = reservationkey2sizecol(prev.key, left=True)
            #             self[left_col] = prev.right
            #         mask = same_level & (r.left != prev.left)
            #         # Left isn't necessarily inclusive -> max both
            #         if mask.any():
            #             max_left = npmax(r.left, prev.left)
            #             self._masked_replace(max_left, prev.left_col, mask)
            #             self._masked_replace(max_left, r.left_col, mask)

            # If the previous reservation has the same level as the next-to-previous
            # level (i.e., the previous level is redundant), we can be put on top of it.
            if i > 1:
                mask = prev.iters_above == prevprev.iters_above
                mask &= prev.iters_above != r.iters_above
                if mask.any():
                    self._shift_reservation_up(prev, prevprev, mask)
                    self._shift_reservation_up(r, prev, mask)
                    _update_changed(mask)

            # If all of our indices == the previous indices (i.e., same level), collapse
            # into the previous level and drop ourselves.
            if i > 0:
                mask = r.iters_above == prev.iters_above
                if mask.all():
                    self._update(prev.right_col, lambda x: npmax(x, r.right))
                    if r.left_col is not None:
                        if prev.left_col is None:
                            left_col = reservationkey2sizecol(prev.key, left=True)
                            self[left_col] = r.left
                        else:
                            self._update(prev.left_col, lambda x: npmax(x, r.left))
                    dropcols = [r.right_col, r.iters_above_col]
                    if r.left_col is not None:
                        dropcols.append(r.left_col)
                    self.drop(columns=dropcols, inplace=True)
                    del datas[i]
                    _update_changed(mask)
                    continue

        return changed.max()

    def _max_right_to_left(self, datas: list[ReservationData]):
        # Set left columns to be max(left, right). Since right may grow & left may not,
        # max(left, right) lower-bounds the maximum size of this level.
        for r in datas:
            if r.left_col is not None:
                self._update(r.left_col, lambda x: npmax(x, r.right))

    def _drop_unneeded_left(self, datas: list[ReservationData]):
        # If the left column is not needed, drop it. Needed means that it has size
        # greater than right in some entries AND, for those entries, it's not duplicated
        # in a higher level
        for i, r in enumerate(datas):
            r = datas[i]
            if r.left_col is not None:
                needed = r.left != r.right
                if i > 0:
                    needed &= r.iters_above != datas[i - 1].iters_above
                if not needed.any():
                    self.drop(columns=[r.left_col], inplace=True)

    def drop_redundant_data(self) -> bool:
        """
        Maxes right -> left to get a maximum size for each level. If there's no
        reservations between the current and previous level, shifts up the current
        level. After shifting everything up as much as possible, drops redundant levels:

        - Left redundant if <= right reservation at the same index, since then they
          couldn't possibly increase usage
        - Level redundant if iterations == previous level's iterations

        Rows must already be sorted
        """
        self.assert_reservations_ordered()
        ncols = len(self.columns)
        nchanges = 0

        for _, datas in self._get_all_reservations().items():
            self._max_right_to_left(datas)

            changed_up_to = len(datas)  # In case we need to shift up multiple times
            while changed_up_to > 0:
                changed_up_to = self._drop_redundant_maybe_shift(datas)
                nchanges += changed_up_to > 0

            self._drop_unneeded_left(datas)

        self.set_reservation_column_numbers()
        self.assert_reservations_ordered()
        self.assert_data_not_redundant()
        return len(self.columns) < ncols or nchanges > 0

    def _shift_reservation_up(
        self,
        src: ReservationData,
        target: ReservationData,
        mask: Series,
        iters_only: bool = False,
    ):
        self._masked_replace(src.iters_above_col, target.iters_above_col, mask)
        if iters_only:
            return
        self._masked_replace(src.right_col, target.right_col, mask)

        # Max the lefts since they're not necessarily inclusive
        if src.left is not None:
            if target.left is None:
                self[reservationkey2sizecol(target.key, left=True)] = target.right
            maxed = npmax(src.left, target.left)
            self._masked_replace(maxed, src.left_col, mask)
            self._masked_replace(maxed, target.left_col, mask)

    def free_to_iters_above(self, iters_above: Series, shift_bottom_left: bool) -> bool:
        """
           A  B
            / | --- 0
           C  D
            / | --- 1  < # iterations above
           E  F
            / | --- 2
           G  H
        ->
           A  B
            / | --- 0
           C  D
             /| --- 1  < # iterations above
        max(E,F,G,H)
        
        If shift_bottom_left is true, the bottom is put on the left & will not live to
        future Einsums.
        """

        # Collapse everything below iters_above
        for _, datas in self._get_all_reservations().items():
            for r in datas:
                mask = r.iters_above > iters_above
                if mask.any():
                    self._update(r.iters_above_col, lambda x: npmin(iters_above, x))
                    
        changed = self.drop_redundant_data()

        # Shift the bottom left
        if shift_bottom_left:
            for _, datas in self._get_all_reservations().items():
                size_above = _reservation_size_above(datas, iters_above)
                for r in datas:
                    mask = r.iters_above == iters_above
                    if not mask.any():
                        continue
                    if r.left_col is None:
                        self[reservationkey2sizecol(r.key, left=True)] = r.right
                    else:
                        self._update(r.left_col, lambda x: npmax(x, r.right))
                    self._masked_replace(size_above, r.right_col, mask)
        return changed

    def alloc_resource(
        self,
        resource: str,
        size: Number | Series,
        n_iters_above: Number | Series,
    ) -> bool:
        need_new = np.ones(len(self), dtype=bool)
        datas = self._get_all_reservations().get(resource, [])
        for r in datas:
            mask = r.iters_above >= n_iters_above
            need_new &= ~(r.iters_above == n_iters_above)
            self._update(r.right_col, lambda x: x + np.where(mask, size, 0))
            if r.left_col is not None:
                mask = r.iters_above > n_iters_above
                self._update(r.left_col, lambda x: x + np.where(mask, size, 0))

        # Need a new column to get an exact n-iters-above for this resource, unless we
        # had an exact match above.
        if need_new.any():
            r = ReservationData(ReservationKey(resource, len(datas), left=False), self)
            parent_size = _reservation_size_at_or_above(datas, n_iters_above)
            last_right = datas[-1].right if datas else 0
            last_iters = datas[-1].iters_above if datas else 0
            self[r.right_col] = np.where(need_new, parent_size + size, last_right)
            self[r.iters_above_col] = np.where(need_new, n_iters_above, last_iters)
            self.sort_reservations()

    def fill_missing_rows(self):
        """
        Rows may only ever be missing @ the end, so just set a super high #iterations
        above and a size of (maximum seen before) for all missing rows.
        """
        cast_low = lambda x: x.fillna(0)
        cast_high = lambda x: x.fillna(float("inf"))

        for _, datas in self._get_all_reservations().items():
            max_size = Series(np.zeros(len(self)), index=self.index)
            for r in datas:
                # Very high #iterations above default
                self._update(r.iters_above_col, cast_high)

                # Low size default, but grab previous if it's there
                self._update(r.right_col, cast_low)
                self._update(r.right_col, lambda x: npmax(max_size, x))
                max_size = npmax(max_size, self[r.right_col])
                if r.left_col is not None:
                    self._update(r.left_col, cast_low)
                    self._update(r.left_col, lambda x: npmax(max_size, x))

    def _update(self, colname: str, f: Callable[[Series], Series]):
        self.loc[:, colname] = f(self[colname])

    def _masked_replace(
        self,
        src: str | Series,
        target: str | Series,
        mask: Series,
    ):
        if isinstance(src, str):
            src = self[src]
        if isinstance(target, str):
            target = self[target]
        self.loc[:, target.name] = np.where(mask, src, target)
