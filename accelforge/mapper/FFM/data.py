"""
Results from mapping exploration.
"""

import pandas as pd

from accelforge.mapper.FFM._pareto_df.df_convention import col2action
from accelforge.util._base_analysis_types import ActionKey, VerboseActionKey


class ResultDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ResultDataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series

    @property
    def actions(self) -> "ResultDataFrame":
        """Returns a ResultDataFrame with all action-related columns."""
        action_columns = [col for col in self.columns if "action" in col]
        return self[[action_columns]]

    @property
    def actions_df(self) -> "ActionDataFrame":
        """Return an ActionDataFrame."""
        df = self.actions
        if any(isinstance(col2action(col), VerboseActionKey) for col in df.columns):
            columns = [
                col
                for col in df.columns
                if isinstance(col2action(col), VerboseActionKey)
            ]

    @property
    def energy(self) -> "ResultDataFrame":
        """Returns a ResultDataFrame with all energy-related columns."""
        action_columns = [col for col in self.columns if "energy" in col]
        return self[[action_columns]]


class ActionDataFrame(pd.DataFrame):
    """
    A hierarchical column dataframe with action counts.
    """

    @property
    def _constructor(self):
        return ResultDataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series


class VerboseActionDataFrame(pd.DataFrame):
    """
    A hierarchical column dataframe with verbose action counts.
    """
