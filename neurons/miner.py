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
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 128
        self.tolerance = 3

        self.config = self.get_config()
        node = SubstrateInterface(url=self.config.network)
        self.epoch_length = node.query("SubtensorModule", "Tempo", [self.config.netuid]).value
        self.setup_logging()
        self.current_block = 0
        self.current_challenge_protein = None
        self.last_challenge_protein = None
        self.psichic_wrapper = PsichicWrapper()
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
        parser.add_argument('--network', default='wss://archive.chain.opentensor.ai:443', help='Network to use')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
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

            # Initialize and sync metagraph
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.metagraph.sync()
            bt.logging.info(f"Metagraph synced: {self.metagraph}")
            
            # Log stake distribution
            stakes = self.metagraph.S.tolist()
            sorted_stakes = sorted([(i, s) for i, s in enumerate(stakes)], key=lambda x: x[1], reverse=True)
            bt.logging.info("Top 5 validators by stake:")
            for uid, stake in sorted_stakes[:5]:
                bt.logging.info(f"UID: {uid}, Stake: {stake}")

    async def get_commitments(self, metagraph, block_hash: str) -> dict:

        """
        Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

        Args:
            subtensor: The subtensor client object.
            netuid (int): The network ID.
            block (int, optional): The block number to query. Defaults to None.

        Returns:
            dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
                block, and decoded commitment data.
        """

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
                bt.logging.info(f'Current block: {current_block}, Epoch length: {self.epoch_length}')
                bt.logging.info(f'Checking block: {current_block - self.epoch_length}')

                # Check if any commitment has been made on the last epoch
                block_to_check = current_block - self.epoch_length
                previous_metagraph = await self.subtensor.metagraph(self.config.netuid, block=block_to_check + self.tolerance)
                block_hash_to_check = await self.subtensor.determine_block_hash(block_to_check + self.tolerance)
                bt.logging.info(f'Using block hash from block: {block_to_check + self.tolerance}')
                
                previous_commitments = await self.get_commitments(previous_metagraph, block_hash=block_hash_to_check)
                bt.logging.info(f'Found {len(previous_commitments)} total commitments')
                
                # More lenient filtering - only check if commitment is from current or previous epoch
                current_epoch = current_block // self.epoch_length
                epoch_commitments = {
                    k: v for k, v in previous_commitments.items() 
                    if v.block // self.epoch_length >= current_epoch - 1
                }
                bt.logging.info(f'Found {len(epoch_commitments)} commitments within epoch window')

                if not epoch_commitments:
                    bt.logging.warning("No commitments found within epoch window")
                    await asyncio.sleep(12)
                    continue

                # Find validator with highest stake among valid commitments
                max_stake = -1
                high_stake_protein_commitment = None
                for commit in epoch_commitments.values():
                    stake = float(previous_metagraph.S[commit.uid])
                    # bt.logging.debug(f"Checking validator {commit.uid} with stake {stake}")
                    if stake > max_stake:
                        max_stake = stake
                        high_stake_protein_commitment = commit

                if not high_stake_protein_commitment:
                    bt.logging.error("Error getting current protein commitment.")
                    current_protein = None
                    continue

                if max_stake <= 0:
                    bt.logging.warning(f"Highest staking validator has 0 or negative stake ({max_stake}). This might indicate a metagraph sync issue.")
                    await self.metagraph.sync()
                    await asyncio.sleep(12)
                    continue

                current_protein = high_stake_protein_commitment.data
                if current_protein is not None:
                    bt.logging.info(f"Current protein: {current_protein} from validator {high_stake_protein_commitment.uid} with stake: {max_stake}")
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
        Continuously runs the PSICHIC model on batches of molecules from the dataset.

        This method streams random chunks of molecule data from a Hugging Face dataset,
        processes them through the PSICHIC model to predict binding affinities, and updates
        the best candidate when a higher scoring molecule is found. Runs in a separate thread
        until the shutdown event is triggered.

        The method:
        1. Streams data in chunks from the dataset
        2. Cleans the product names and SMILES strings
        3. Runs PSICHIC predictions on each chunk
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
                    # Run the PSICHIC model on the chunk.
                    bt.logging.debug(f'Running inference...')
                    chunk_psichic_scores = self.psichic_wrapper.run_validation(df['product_smiles'].tolist())
                    chunk_psichic_scores = chunk_psichic_scores.sort_values(by=self.psichic_result_column_name, ascending=False).reset_index(drop=True)
                    if chunk_psichic_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        async with self.shared_lock:
                            candidate_molecule = chunk_psichic_scores['Ligand'].iloc[0]
                            self.best_score = chunk_psichic_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                            bt.logging.info(f"New best score: {self.best_score}, New candidate product: {self.candidate_product}")
                        await asyncio.sleep(1)
                    await asyncio.sleep(3)

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
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

                    # Initialize PSICHIC for new protein
                    bt.logging.info(f'Initializing model for protein sequence: {protein_sequence}')
                    try:
                        self.psichic_wrapper.run_challenge_start(protein_sequence)
                        bt.logging.info('Model initialized successfully.')
                    except Exception as e:
                        try:
                            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
                            self.psichic_wrapper.run_challenge_start(protein_sequence)
                            bt.logging.info('Model initialized successfully.')
                        except Exception as e:
                            bt.logging.error(f'Error initializing model: {e}')

                    # Start inference loop
                    try:
                        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                        bt.logging.debug(f'Inference task started successfully')
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
                if self.current_block % 60 == 0:
                    await self.metagraph.sync()
                    log = (
                        f'Block: {self.metagraph.block.item()} | '
                        f'Number of nodes: {self.metagraph.n} | '
                        f'Current epoch: {self.metagraph.block.item() // self.epoch_length}'
                    )
                    bt.logging.info(log)
                self.current_block += 1

            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                exit()

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())
