"""
Data loading and validation module.

Loads neural spike data from Parquet format and validates structure
according to Rumyantsev et al. 2020 dataset specifications.
"""

from pathlib import Path
from typing import Union

import polars as pl


class DataLoader:
    """
    Load and validate neural dataset from parquet.
    
    CRITICAL: cell_idx is per-mouse indexed (not globally unique).
    Always process data per-mouse to avoid ID collisions.
    
    Paper Reference:
    Rumyantsev et al., "Fundamental bounds on the fidelity of sensory 
    cortical coding", Nature 2020, Methods pages 6-13.
    
    Dataset contains:
    - 5 mice with ~1000-2000 cells each (total: 8,029)
    - ~350 trials per stimulus (±30° gratings)
    - 14 time bins at 0.275s resolution
    """
    
    # ACTUAL column names in the parquet file
    REQUIRED_COLUMNS = [
        'mouse_id',
        'cell_idx',
        'trial_idx',
        'sample_idx',
        'behavior',
        'amplitude'
    ]
    
    def __init__(self, parquet_path: Union[str, Path]):
        """
        Initialize loader and validate data structure.
        
        Args:
            parquet_path: Path to parquet file with neural data
            
        Raises:
            ValueError: If required columns missing or data invalid
        """
        self.parquet_path = Path(parquet_path)
        self._load_and_validate()
    
    def _load_and_validate(self) -> None:
        """
        Load data and check structure.
        
        Paper: Data from 5 mice, 8,029 neurons total (Methods, page 6)
        """
        self.data = pl.read_parquet(self.parquet_path)
        
        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Cache statistics
        self.n_mice = self.data['mouse_id'].n_unique()
        
        # NOTE: Don't use data['cell_idx'].n_unique() - gives wrong count!
        # cell_idx is per-mouse indexed, must count separately per mouse
        
        self.n_total_trials = self.data['trial_idx'].n_unique()
        
    def count_total_cells(self) -> int:
        """
        Count total cells across all mice.
        
        CRITICAL: cell_idx is per-mouse indexed, so we must count
        unique cells per mouse and sum.
        
        Paper: "8,029 cells from 5 mice" (Methods, page 6)
        
        Returns:
            Total number of unique cells across all mice
        
        Example:
            >>> loader = DataLoader('data.parquet')
            >>> total = loader.count_total_cells()
            >>> print(f"Total neurons: {total}")
            Total neurons: 8029
        """
        cells_per_mouse = (
            self.data
            .group_by('mouse_id')
            .agg(pl.col('cell_idx').n_unique().alias('n_cells'))
        )
        return cells_per_mouse['n_cells'].sum()
        
    def get_mouse_data(self, mouse_id: str) -> pl.DataFrame:
        """
        Extract data for a single mouse.
        
        Paper: Data collected from 5 mice (Methods, page 6)
        
        Args:
            mouse_id: Mouse identifier (e.g., "Mouse_L347")
            
        Returns:
            DataFrame filtered to single mouse
            
        Example:
            >>> loader = DataLoader('data.parquet')
            >>> mouse_data = loader.get_mouse_data('Mouse_L347')
            >>> print(f"Cells: {mouse_data['cell_idx'].n_unique()}")
            Cells: 1921
        """
        return self.data.filter(pl.col('mouse_id') == mouse_id)
    
    def get_cell_data(self, mouse_id: str, cell_idx: int) -> pl.DataFrame:
        """
        Extract all trials for a specific cell.
        
        Args:
            mouse_id: Mouse identifier
            cell_idx: Cell index (unique within mouse)
            
        Returns:
            DataFrame with all data for this cell
            
        Example:
            >>> loader = DataLoader('data.parquet')
            >>> cell_data = loader.get_cell_data('Mouse_L347', 0)
            >>> print(f"Trials: {cell_data['trial_idx'].n_unique()}")
            Trials: 662
        """
        return self.data.filter(
            (pl.col('mouse_id') == mouse_id) & 
            (pl.col('cell_idx') == cell_idx)
        )
    
    def add_stimulus_labels(self) -> pl.DataFrame:
        """
        Add stimulus_type column mapping behavior codes to labels.
        
        Paper: "Two drifting grating stimuli (±30° from vertical)"
        (Methods, page 6)
        
        Maps: 
        - 30 → 'A' (+30° grating)
        - -30 → 'B' (-30° grating)
        
        Returns:
            DataFrame with added 'stimulus_type' column
            
        Example:
            >>> loader = DataLoader('data.parquet')
            >>> data = loader.add_stimulus_labels()
            >>> print(data.select(['behavior', 'stimulus_type']).unique())
            shape: (2, 2)
            ┌──────────┬───────────────┐
            │ behavior │ stimulus_type │
            │ ---      │ ---           │
            │ i16      │ str           │
            ╞══════════╪═══════════════╡
            │ 30       │ A             │
            │ -30      │ B             │
            └──────────┴───────────────┘
        """
        stimulus_map = {30: 'A', -30: 'B'}
        return self.data.with_columns(
            pl.col('behavior').map_dict(stimulus_map).alias('stimulus_type')
        )

