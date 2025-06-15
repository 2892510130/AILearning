"""
Dataset
"""

import os
import numpy as np
import pandas as pd
import lightning as L
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split

def read_data_ml100k(data_dir:str="../Data/ml-100k") -> pd.DataFrame:
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

class ML100kData(Dataset):
    def __init__(self, data_dir:str="../Data/ml-100k", normalize_rating:bool=False):
        self.data_dir = data_dir
        self.normalize_rating = normalize_rating
        self.df, self.num_users, self.num_items = read_data_ml100k(data_dir)
        self.user_id = self.df.user_id.values - 1
        self.item_id = self.df.item_id.values - 1
        self.rating = self.df.rating.values.astype(np.float32)
        
    def split(self, train_ratio=0.8):
        train_len = int(train_ratio * len(self))
        test_len = len(self) - train_len
        return random_split(self, [train_len, test_len])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx:int):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]

class LitData(L.LightningDataModule):
    def __init__(
        self, 
        dataset:Dataset, 
        train_ratio:float=0.8, 
        batch_size:int=32, 
        num_workers:int=4,
        prefetch_factor:int=4
    ):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": True if num_workers > 0 else False,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None
        }
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.num_users = getattr(self.dataset, "num_users", None)
        self.num_items = getattr(self.dataset, "num_items", None)

    def setup(self, stage:str):
        self.train_split, self.test_split = self.dataset.split(
            self.train_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

class AutoRecData(ML100kData):
    def __init__(self, data_dir:str="../Data/ml-100k", user_based=False, normalize_rating:bool=False):
        super().__init__(data_dir, normalize_rating)
        self.user_based = user_based
        self.rating_matrix = np.zeros(
            (self.num_items, self.num_users), dtype=np.float32)
        self.rating_matrix[self.item_id, self.user_id] = self.rating
        if normalize_rating:
            self.rating_matrix /= 5.0

    def __len__(self):
        if self.user_based:
            return self.num_users
        else:
            return self.num_items

    def __getitem__(self, idx:int):
        if self.user_based:
            return self.rating_matrix[:, idx]
        else:
            return self.rating_matrix[idx]

class LitAutoRecData(L.LightningDataModule):
    def __init__(
        self, 
        dataset:Dataset, 
        train_ratio:float=0.8, 
        batch_size:int=32, 
        num_workers:int=4
    ):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": True if num_workers > 0 else False
        }
        self._log_hyperparams = True
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.num_users = getattr(self.dataset, "num_users", None)
        self.num_items = getattr(self.dataset, "num_items", None)

    def setup(self, stage:str):
        # self.num_users = getattr(self.dataset, "num_users", None)
        # self.num_items = getattr(self.dataset, "num_items", None)
        self.train_split, self.test_split = self.dataset.split(
            self.train_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

class ML100KPairWise(ML100kData):
    def __init__(self, data_dir="../Data/ml-100k",
                 test_leave_out=1,
                 test_sample_size: int = None):
        """Pair Wise loader to train NeuMF model.
        Samples are slightly different based on train/test mode.

        In training mode:
        - user_id: int
        - item_id: int
            Item id that user has interacted with
        - neg_item: int
            Item id that user hasn't interacted with while training

        In testing mode:
        - user_id: int
        - item_id: int
            Random item_id to be ranked by the model
        - is_pos: bool
            If True, this item is a positive item 
            that user has interacted with in groundtruth data.


        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./ml-100k"
        test_leave_out : int, optional
            Leave out how many items per user for testing
            By default 1
        test_sample_size : int, optional
            It is time-consuming to rank all items for every user during
            evaluation, we can randomly choose a subset of items to rank
            If None, rank all items.
            By default None
        """
        super().__init__(data_dir)
        # Use 0-based indexing consistently
        self.set_all_item_ids = set(range(self.num_items))  # 0 to num_items-1
        self.test_leave_out = test_leave_out
        self.test_sample_size = test_sample_size
        # general
        self.train = None
        self.has_setup = False
        # Split Dataframe
        self.split_dataframe()
        self.build_candidates()

    def split_dataframe(self):
        """Split ML100K dataframe with the strategy leave-n-out
        with timestamp order.
        """
        user_group = self.df.groupby("user_id", sort=False)
        train_df = []
        test_df = []
        for user_id, user_df in user_group:
            user_df = user_df.sort_values("timestamp")
            train_df.append(user_df[:-self.test_leave_out])
            test_df.append(user_df[-self.test_leave_out:])
        self.train_df = pd.concat(train_df)
        self.test_df = pd.concat(test_df)

    def build_candidates(self):
        # Train - Make both user_id and item_id 0-based
        self.observed_items_per_user_in_train = {
            int(user_id) - 1: user_df.item_id.values - 1
            for user_id, user_df in self.train_df.groupby("user_id", sort=False)
        }
        self.unobserved_items_per_user_in_train = {
            user_id: np.array(
                list(self.set_all_item_ids - set(observed_items)))
            for user_id, observed_items in self.observed_items_per_user_in_train.items()
        }
        # Test - Make both user_id and item_id 0-based
        self.gt_pos_items_per_user_in_test = {
            int(user_id) - 1: user_df[-self.test_leave_out:].item_id.values - 1
            for user_id, user_df in self.test_df.groupby("user_id", sort=False)
        }

    def split(self, *args, **kwargs):
        # Train split
        train_split = deepcopy(self)
        train_split.user_id = self.train_df.user_id.values - 1
        train_split.item_id = self.train_df.item_id.values - 1
        train_split.train = True
        train_split.has_setup = True
        
        # Test split
        test_split = deepcopy(self)
        test_split.user_id = []
        test_split.item_id = []
        for user_id, items in self.unobserved_items_per_user_in_train.items():
            if self.test_sample_size is None:
                sample_items = items
            elif isinstance(self.test_sample_size, int):
                sample_items = np.random.choice(items, self.test_sample_size)
            else:
                raise TypeError("self.test_sample_size should be int")
            sample_items = np.concatenate(
                [test_split.gt_pos_items_per_user_in_test[user_id],
                 sample_items])
            sample_items = np.unique(sample_items)
            test_split.user_id += [user_id]*len(sample_items)
            test_split.item_id.append(sample_items)
        test_split.user_id = np.array(test_split.user_id)
        test_split.item_id = np.concatenate(test_split.item_id)
        test_split.train = False
        test_split.has_setup = True
        return train_split, test_split

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        assert self.has_setup, "Must run self.setup()"
        if self.train:
            user_id = self.user_id[idx]
            pos_item = self.item_id[idx]
            neg_item = np.random.choice(
                self.unobserved_items_per_user_in_train[int(user_id)])
            return user_id, pos_item, neg_item
        else:
            user_id = self.user_id[idx]
            item_id = self.item_id[idx]
            is_pos = item_id in self.gt_pos_items_per_user_in_test[user_id]
            return user_id, item_id, is_pos

class ML100KSequence(ML100KPairWise):
    def __init__(self, data_dir="../Data/ml-100k",
                 test_leave_out=1,
                 test_sample_size=100,
                 seq_len=5):
        """Sequence data to train Caser model
        Similarly to Pair Wise dataset, the sample depends on train/test mode.

        In training mode:
        - user_id: int
        - seq: List[int]
            Sequence of last N item ids that user has interacted with.
        - target_item: int
            Target item id that user will interact with after the sequence
        - neg_item: int
            Item id that user doesn't interacted with while training

        In testing mode:
        - user_id: int
        - seq: List[int]
            Sequence of last N item ids that user has interacted with.
        - item_id: int
            Random item_id to be ranked by the model
        - is_pos: bool
            If True, this item is a positive item 
            that user has interacted with in groundtruth data.

        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./ml-100k"
        test_leave_out : int, optional
            Leave out how many items per user for testing
            By default 1
        test_sample_size : int, optional
            It is time-consuming to rank all items for every user during
            evaluation, we can randomly choose a subset of items to rank
            If None, rank all items.
            By default None
        seq_len : int, optional
            Length of sequence of item ids, by default 5
        """
        self.seq_len = seq_len
        super().__init__(data_dir, test_leave_out, test_sample_size)
        self.getitem_df = None

    def split_dataframe(self):
        """Split dataframe ensuring users have enough interactions for sequences"""
        user_group = self.df.groupby("user_id", sort=False)
        train_df = []
        test_df = []
        
        for user_id, user_df in user_group:
            user_df = user_df.sort_values("timestamp")
            
            # Skip users with insufficient interactions
            # Need at least seq_len + test_leave_out interactions
            if len(user_df) < self.seq_len + self.test_leave_out:
                continue
                
            train_df.append(user_df[:-self.test_leave_out])
            test_df.append(user_df[-(self.test_leave_out + self.seq_len):])
            
        self.train_df = pd.concat(train_df) if train_df else pd.DataFrame()
        self.test_df = pd.concat(test_df) if test_df else pd.DataFrame()

    def split(self, *args, **kwargs):
        # Train split
        train_split = deepcopy(self)
        df = []
        
        for _, user_df in self.train_df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp").reset_index(drop=True)
            
            # Skip if not enough data for sequences
            if len(user_df) <= self.seq_len:
                continue
                
            user_id = user_df.user_id.iloc[0] - 1  # Convert to 0-based
            
            # Create sequences and targets
            sequences = []
            targets = []
            
            for i in range(len(user_df) - self.seq_len):
                seq = user_df.item_id.iloc[i:i+self.seq_len].values - 1  # 0-based
                target = user_df.item_id.iloc[i+self.seq_len] - 1  # 0-based
                sequences.append(seq)
                targets.append(target)
            
            if sequences:  # Only add if we have valid sequences
                df.append(pd.DataFrame({
                    "user_id": [user_id] * len(sequences),
                    "seq": sequences,
                    "target_item": targets
                }))
        
        if df:
            train_split.getitem_df = pd.concat(df, ignore_index=True)
        else:
            train_split.getitem_df = pd.DataFrame(columns=["user_id", "seq", "target_item"])
        train_split.train = True

        # Test split
        test_split = deepcopy(self)
        df = []
        
        for uid, user_df in self.test_df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp").reset_index(drop=True)
            uid_0based = uid - 1  # Convert to 0-based
            
            # Skip if user not in training data (causes KeyError)
            if uid_0based not in self.unobserved_items_per_user_in_train:
                continue
                
            # Skip if not enough data for sequences
            if len(user_df) <= self.seq_len:
                continue
            
            # Create sequences and targets for test
            sequences = []
            targets = []
            
            for i in range(len(user_df) - self.seq_len):
                seq = user_df.item_id.iloc[i:i+self.seq_len].values - 1  # 0-based
                target = user_df.item_id.iloc[i+self.seq_len] - 1  # 0-based
                sequences.append(seq)
                targets.append(target)
            
            if not sequences:  # Skip if no valid sequences
                continue
            
            # Sample negative items
            unobserved_items = self.unobserved_items_per_user_in_train[uid_0based]
            
            if len(unobserved_items) == 0:
                continue
                
            # Fix the np.concatenate issue - remove unnecessary wrapping
            sample_size = min(self.test_sample_size, len(unobserved_items))
            unobserved_item_id = np.random.choice(
                unobserved_items,
                sample_size,
                replace=False
            )
            
            # Create test samples for each sequence
            all_user_ids = []
            all_seqs = []
            all_item_ids = []
            all_is_pos = []
            
            for seq, target in zip(sequences, targets):
                # Combine target with negative samples
                test_items = np.unique(np.append(unobserved_item_id, target))
                
                # Create entries for this sequence
                seq_user_ids = [uid_0based] * len(test_items)
                seq_seqs = [seq] * len(test_items)
                seq_is_pos = [item == target for item in test_items]
                
                all_user_ids.extend(seq_user_ids)
                all_seqs.extend(seq_seqs)
                all_item_ids.extend(test_items)
                all_is_pos.extend(seq_is_pos)
            
            if all_user_ids:  # Only add if we have valid data
                df.append(pd.DataFrame({
                    "user_id": all_user_ids,
                    "seq": all_seqs,
                    "item_id": all_item_ids,
                    "is_pos": all_is_pos
                }))
        
        if df:
            test_split.getitem_df = pd.concat(df, ignore_index=True)
        else:
            test_split.getitem_df = pd.DataFrame(columns=["user_id", "seq", "item_id", "is_pos"])
        test_split.train = False
        
        return train_split, test_split

    def __len__(self):
        assert self.getitem_df is not None
        return len(self.getitem_df)

    def __getitem__(self, idx):
        assert self.getitem_df is not None
        row = self.getitem_df.iloc[idx]
        
        if self.train:
            user_id = int(row.user_id)
            neg_item = np.random.choice(
                self.unobserved_items_per_user_in_train[user_id])
            return user_id, row.seq, row.target_item, neg_item
        else:
            return row.user_id, row.seq, row.item_id, row.is_pos
        

def csv_reader(data_path):
    dtype = {i: str for i in range(1, 35)}
    dtype[0] = np.uint8
    return pd.read_csv(data_path, sep="\t",
                       header=None, dtype=dtype)

class ConstantFactory:
    def __init__(self, value):
        self.value = value
    
    def __call__(self):
        return self.value

def _constant_factory(v):
    return ConstantFactory(v)

class CTRDataset(Dataset):
    def __init__(self, data_dir="../Data/ctr", min_threshold=4):
        """Read CTR dataset from train.csv and test.csv

        Parameters
        ----------
        data_dir : str
            Path to directory containing train.csv and test.csv
        min_threshold : int, optional
            Remove feature values that occurs less than this threshold
            By default 4
        """
        # Read csv
        self.data_dir = data_dir
        self.train_df = csv_reader(os.path.join(data_dir, "train.csv"))
        self.test_df = csv_reader(os.path.join(data_dir, "test.csv"))
        self.feat_cols = [i for i in range(1, len(self.train_df.columns))]
        # Count unique values in each columns
        feat_counts = {
            col: self.train_df[col].value_counts()
            for col in self.feat_cols}
        # Feature mapper maps a unique encoded value to an identifier
        # So each value is considered to be a categorical value.
        # Unique values are filtered with occurence greater or equal to min_threshold
        # A default value will be assign to values that not defined in feature mapper
        self.feat_mapper = {}

        for col, val_counts in feat_counts.items():
            val = val_counts.index[val_counts >= min_threshold]
            default = len(val)
            self.feat_mapper[col] = pd.Series(
                range(len(val)), index=val, dtype=np.int32
            ).to_dict(into=defaultdict(_constant_factory(default)))
        # Feature dimension = number of unique values = number of values in mapper + defaults
        self.feat_dims = np.array([len(mapper) + 1
                                   for mapper in self.feat_mapper.values()])
        # Offset is a value add to the whole field to discriminate values in different columns
        self.offsets = np.array((0, *np.cumsum(self.feat_dims).tolist()[:-1])).astype(np.int32)
        # Map values in dataframe
        for col, mapper in self.feat_mapper.items():
            self.train_df[col] = self.train_df[col].map(mapper)
            self.test_df[col] = self.test_df[col].map(mapper)
        # For each split
        self.X = None
        self.y = None

    def build_items(self, train=True):
        if train:
            df = self.train_df
        else:
            df = self.test_df
        self.X = df[self.feat_cols].values + self.offsets
        self.y = df[0].values

    def split(self, *args, **kwargs) -> tuple[Dataset, Dataset]:
        train_split = deepcopy(self)
        train_split.build_items(True)

        test_split = deepcopy(self)
        test_split.build_items(False)

        return train_split, test_split

    def __len__(self):
        assert self.X is not None and self.y is not None
        return len(self.X)

    def __getitem__(self, idx):
        assert self.X is not None and self.y is not None
        return self.X[idx], self.y[idx]