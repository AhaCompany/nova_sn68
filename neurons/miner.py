import os
import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace
import sys

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from DiffDock.wrapper import DiffDockWrapper

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 128
        self.epoch_length = 360

        self.config = self.get_config()
        self.setup_logging()
        self.current_block = 0
        self.current_challenge_protein = None
        self.last_challenge_protein = None
        
        # Initialize model wrappers based on command line arguments
        self.use_diffdock = self.config.use_diffdock if hasattr(self.config, 'use_diffdock') else False
        
        # Log which model is being used
        if self.use_diffdock:
            bt.logging.info("Using DiffDock model for binding predictions")
        else:
            bt.logging.info("Using PSICHIC model for binding predictions")
            
        self.psichic_wrapper = PsichicWrapper()
        self.diffdock_wrapper = DiffDockWrapper()
        
        self.candidate_product = None
        self.candidate_product_score = 0
        self.best_score = 0
        self.last_submitted_product = None
        self.shared_lock = asyncio.Lock()
        self.inference_task = None
        self.shutdown_event = asyncio.Event()

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # Adds override arguments for network.
        parser.add_argument('--network', default='finney', help='Network to use')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        # Add argument to choose between PSICHIC and DiffDock
        parser.add_argument('--use_diffdock', action='store_true', help="Use DiffDock model instead of PSICHIC")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Parse the config.
        config = bt.config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    async def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        async with bt.async_subtensor(network=self.config.network) as subtensor:
            self.subtensor = subtensor
            bt.logging.info(f"Subtensor: {self.subtensor}")

            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            bt.logging.info(f"Metagraph: {self.metagraph}")

    async def get_commitments(self, block: int = None) -> dict:
        """
        Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

        Args:
            block (int, optional): The block number to query. Defaults to None.

        Returns:
            dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
                block, and decoded commitment data.
        """
        # Use the provided netuid to fetch the corresponding metagraph.
        metagraph = await self.subtensor.metagraph(self.config.netuid)
        # Determine the block hash if a block is specified.
        block_hash = await self.subtensor.determine_block_hash(block) if block is not None else None

        # Gather commitment queries for all validators (hotkeys) concurrently.
        commits = await asyncio.gather(*[
            self.subtensor.substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.config.netuid, hotkey],
                block_hash=block_hash,
            ) for hotkey in metagraph.hotkeys
        ])

        # Process the results and build a dictionary with additional metadata.
        result = {}
        for uid, hotkey in enumerate(metagraph.hotkeys):
            commit = cast(dict, commits[uid])
            if commit:
                result[hotkey] = SimpleNamespace(
                    uid=uid,
                    hotkey=hotkey,
                    block=commit['block'],
                    data=decode_metadata(commit)
                )
        return result

    async def get_current_challenge_protein(self):
        while True:
            try:
                current_block = await self.subtensor.get_current_block()
                bt.logging.debug(f'Current block: {current_block}')

                # Check if any commitment has been made on the last epoch
                prev_epoch = current_block - self.epoch_length
                previous_metagraph = await self.subtensor.metagraph(self.config.netuid, block=current_block)
                previous_commitments = await self.get_commitments(block=current_block)
                #print(previous_commitments)

                # Determine the current protein as that set by the validator with the highest stake.
                best_stake = -math.inf
                current_protein = None
                for index, (hotkey, commit) in enumerate(previous_commitments.items()):
                    if current_block - commit.block <= self.epoch_length:
                        # Access the stake for the given hotkey from the metagraph.
                        hotkey_stake = previous_metagraph.S[index]
                        # Choose the commitment with the highest stake and valid data.
                        if hotkey_stake > best_stake and commit.data is not None:
                            best_stake = hotkey_stake
                            current_protein = commit.data

                    if current_protein is not None:
                        bt.logging.info(f'Challenge protein: {current_protein}')
                        return current_protein
                    else:
                        bt.logging.info(f'No protein set yet. Waiting...')
                        await asyncio.sleep(12)

            except Exception as e:
                bt.logging.error(f'Error getting challenge protein: {e}')
                break

    def stream_random_chunk_from_dataset(self):
        # Streams a random chunk from the dataset repo on huggingface.
        files = list_repo_files(self.hugging_face_dataset_repo, repo_type='dataset')
        files = [file for file in files if file.endswith('.csv')]
        random_file = random.choice(files)
        dataset_dict = load_dataset(self.hugging_face_dataset_repo,
                                    data_files={'train': random_file},
                                    streaming=True,
                                    )
        dataset = dataset_dict['train']
        batched = dataset.batch(self.chunk_size)
        return batched

    async def run_psichic_model_loop(self):
        """
        Continuously runs the binding affinity model (PSICHIC or DiffDock) on batches of molecules.

        This method streams random chunks of molecule data from a Hugging Face dataset,
        processes them through the selected model to predict binding affinities, and updates
        the best candidate when a higher scoring molecule is found. Runs in a separate thread
        until the shutdown event is triggered.

        The method:
        1. Streams data in chunks from the dataset
        2. Cleans the product names and SMILES strings
        3. Runs model predictions on each chunk
        4. Updates the best candidate if a higher score is found
        5. Continues until shutdown_event is set

        Raises:
            Exception: Logs any errors during execution and sets the shutdown event
        """
        dataset = self.stream_random_chunk_from_dataset()
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    df = pd.DataFrame.from_dict(chunk)
                    df['product_name'] = df['product_name'].apply(lambda x: x.replace('"', ''))
                    df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))
                    
                    # Run the selected model on the chunk
                    bt.logging.debug(f'Running inference...')
                    if self.use_diffdock:
                        # Run DiffDock model
                        chunk_scores = self.diffdock_wrapper.run_validation(df['product_smiles'].tolist())
                    else:
                        # Run PSICHIC model
                        chunk_scores = self.psichic_wrapper.run_validation(df['product_smiles'].tolist())
                    
                    # Sort results by predicted binding affinity
                    chunk_scores = chunk_scores.sort_values(by=self.psichic_result_column_name, ascending=False).reset_index(drop=True)
                    
                    # Check if we found a better candidate
                    if chunk_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        async with self.shared_lock:
                            candidate_molecule = chunk_scores['Ligand'].iloc[0]
                            self.best_score = chunk_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                            
                            # Log which model found the new best candidate
                            model_name = "DiffDock" if self.use_diffdock else "PSICHIC"
                            bt.logging.info(f"New best score from {model_name}: {self.best_score}, New candidate product: {self.candidate_product}")
                        
                        await asyncio.sleep(1)
                    await asyncio.sleep(3)

            except Exception as e:
                model_name = "DiffDock" if self.use_diffdock else "PSICHIC"
                bt.logging.error(f"Error running {model_name} model: {e}")
                self.shutdown_event.set()

    async def run(self):
        # The Main Mining Loop.
        bt.logging.info("Starting miner loop.")
        await self.setup_bittensor_objects()
        while True:
            try:
                self.current_challenge_protein = await self.get_current_challenge_protein()

                # Check if protein has changed
                if self.current_challenge_protein != self.last_challenge_protein:
                    bt.logging.info(f'Got new challenge protein: {self.current_challenge_protein}')
                    self.last_challenge_protein = self.current_challenge_protein

                    # If old task still running, set shutdown event
                    if self.inference_task:
                        if not self.inference_task.done():
                            self.shutdown_event.set()
                            bt.logging.debug(f"Shutdown event set for old inference task.")

                            # reset old values for best score, etc
                            self.candidate_product = None
                            self.candidate_product_score = 0
                            self.best_score = 0
                            self.last_submitted_product = None
                            self.shutdown_event = asyncio.Event()

                    # Get protein sequence from uniprot
                    protein_sequence = get_sequence_from_protein_code(self.current_challenge_protein)

                    # Initialize model for new protein
                    bt.logging.info(f'Initializing model for protein sequence: {protein_sequence}')
                    try:
                        if self.use_diffdock:
                            self.diffdock_wrapper.run_challenge_start(protein_sequence)
                            bt.logging.info('DiffDock model initialized successfully.')
                        else:
                            self.psichic_wrapper.run_challenge_start(protein_sequence)
                            bt.logging.info('PSICHIC model initialized successfully.')
                    except Exception as e:
                        bt.logging.error(f'Error initializing model: {e}')

                    # Start inference loop
                    try:
                        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                        model_name = "DiffDock" if self.use_diffdock else "PSICHIC"
                        bt.logging.debug(f'{model_name} inference task started successfully')
                    except Exception as e:
                        bt.logging.error(f'Error initializing inference: {e}')


                # Check if candidate product has changed
                async with self.shared_lock:
                    if self.candidate_product:
                        if self.candidate_product != self.last_submitted_product:
                            current_product_to_submit = self.candidate_product
                            current_product_score = self.best_score
                            try:
                                await self.subtensor.set_commitment(
                                    wallet=self.wallet,
                                    netuid=self.config.netuid,
                                    data=current_product_to_submit
                                    )
                                self.last_submitted_product = current_product_to_submit
                                bt.logging.info(f'Submitted product: {current_product_to_submit} with score: {current_product_score}')

                            except MetadataError as e:
                                bt.logging.info(f'Too soon to commit again, will keep looking for better candidates.')
                            except Exception as e:
                                bt.logging.error(e)
                await asyncio.sleep(1)

                # Periodically update our knowledge of the network graph.
                #if step % 60 == 0:
                #    await self.metagraph.sync()
                #    log = (
                #        f'Block: {self.metagraph.block.item()} | '
                #        f'Incentive: {self.metagraph.I[self.my_subnet_uid]} | '
                #    )
                #    bt.logging.info(log)
                #step += 1
                #asyncio.sleep(1)


            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()

# Run the validator.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())
