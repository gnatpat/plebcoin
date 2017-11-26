"""TODO(npat): DO NOT SUBMIT without one-line documentation for node.

TODO(npat): DO NOT SUBMIT without a detailed description of node.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import collections
import hashlib
import multiprocessing
import os
import time

import cryptography
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

NULL_BLOCK_HASH = '0'

#TODO(npat): make this a more permanent solution
class Logging(object):

  @staticmethod
  def info(msg, *args):
    print(msg % args)


logging = Logging()


class BlockHeader(object):

  def __init__(self, prev_hash, merkle_root, nonce):
    """Constructor.

    Args:
      prev_hash: string. SHA256 hash of the block previous to this one.
      merkle_root: string. SHA256 merkle root of all transactions in this block.
      nonce: int. Value added to the hash to prove work has been done.
    """
    self.prev_hash = prev_hash
    self.merkle_root = merkle_root
    self.nonce = nonce

  def Hash(self):
    """Return the hash of this BlockHeader, and thus the hash of the block.

    If we cared about efficiency, it would probably be a good idea to cache
    the hash and only recompute it when something changes, but for now this
    is good enough, and clear.
    """
    m = hashlib.sha256()
    m.update(prev_hash)
    m.update(merkle_root)
    m.update(nonce)
    return m.hexdigest()

  def IsValid(self, difficulty=4):
    # Forcing difficulty to be divisible by 4 means we can just check that the
    # hex representation of the hash starts with d/4 zeroes.
    if difficulty % 4 == 0:
      raise ValueError("Difficulty must be divisible by four.")
    return self.Hash().startswith('0'*(difficulty/4))


TransactionInput = collections.namedtuple("TransactionInput",
                                          ["sender", "transaction"])
""" TransactionInput. One input to a transaction.
  sender: string. Public Key of the sender, represented as a hexstring.
  transaction: string. SHA256 hash of transaction the sender wishes to use
    to pay for this transaction.
"""

TransactionOutput = collections.namedtuple("TransactionOutput",
                                           ["recipient", "amount"])
""" TransactionOutput. One output of a transaction.
  recipient: string. Public Key of the recipient, represented as a
    hexstring.
  amount: float. Amount of PlebCoin the recipient recieves from this
    transaction.
"""

def Transaction(object):

  def __init__(self, inputs, outputs, signatures, is_reward=False):
    """Constructor.

    Args:
      input: list of TransactionInput.
      outputs: list of TransactionOutput.
      signatures: list of string. Signatures, one per input, that verify this
        transaction.
      is_reward: bool. If this transaction is a reward for mining a block.
    """
    self.inputs = inputs
    self.outputs = outputs
    self.signatures = signatures

  def Hash(self):
    m = hashlib.sha256()
    for inp in self.inputs:
      m.update(inp.sender)
      m.update(inp.transaction)
    for output in transaction.outputs:
      m.update(output.receiver)
      m.update(output.amount.hex())
    return m.hexdigest()


  def Verify(self, transactions, used_transactions):
    value_in = 0

    for inp in transaction.inputs:
      prev_transaction_hash = inp.transaction
      if prev_transaction_hash not in transactions:
        return False
      if (inp.sender, prev_transaction_hash) in used_transactions:
        return False

      prev_transaction = transactions[prev_transaction_hash]
      for output in prev_transaction.outputs:
        if output.receiver == tinput.sender:
          value_in += output.amount
          break
      else:
        # The sender did not have a part in the previous transaction
        return False

    if self.is_reward:
      #TODO(npat): Make this reward flexible.
      value_in += 50

    value_out = sum([output.amount for output in transaction.outputs])
    if value_in != value_out:
      return False

    hash_bytes = binascii.unhexlify(self.Hash())

    backend = backends.default_backend()
    prehashed = utils.Prehashed(hashes.SHA256())
    for signature, tinput in zip(transaction.signatures, transaction.inputs):
      signature_bytes = binascii.unhexlify(signature)

      public_key_bytes = binascii.unhexlify(tinput.sender)
      public_key = load_pem_public_key(public_key_bytes, backend)

      try:
        public_key.verify(signature_bytes, hash_bytes, prehashed)
      except cryptography.exceptions.InvalidKey:
        return False

    return True



BlockMetadata = collections.namedtuple('BlockMetadata', ['length', 'timestamp'])

class BlockChainMap(object):
  """Map of blocks to values.

  Allows efficient retrieval of the value stored at the head."""

  def __init__(self, empty_value=None):
    self.head = NULL_BLOCK_HASH
    self.head_metadata = BlockMetadata(0, 0)
    self.values = {NULL_BLOCK_HASH: empty_value}
    self.metadata = {NULL_BLOCK_HASH: self.head_metadata}

  def __setitem__(self, block, value):
    prev_hash = block.header.prev_hash
    if prev_hash not in self.values:
      raise ValueError("Don't have this block's parent.")
    block_hash = _HashBlock(block)
    self.blocks[block_hash] = value

    prev_metadata = self._metadata[prev_hash]
    # We store time backwards for easier comparison.
    metadata = BlockMetadata(prev_metadata.length+1, -time.time())
    if metadata > self.head_metadata:
      self.head = block_hash
      self.head_metadata = metadata

  def __getitem__(self, block):
    block_hash = _HashBlock(block)
    if block_hash not in self.values:
      return ValueError("Don't have any data for this block:\n %s", block_hash)
    return self.values[block_hash]

  def GetHeadValue(self):
    return self.values[self.head]




def _FindMerkleRoot(hashes):
  if len(hashes) == 1:
    return hashes[0]
  next_hashes = []
  while len(hashes)>1:
    m = hashlib.sha256()
    m.update(hashes.pop())
    m.update(hashes.pop())
    next_hashes.append(m.hexdigest())
  if hashes:
    next_hashes.append(hashes.pop)
  return _FindMerkleRoot(next_hashes)



def _MineBlock(block):
  for nonce in xrange(100000000):
    block.header.nonce = nonce
    if _BlockHashIsCorrect(_HashBlock(block)):
      return block


def _CreateBlock(transactions, prev_hash, length):
  """Create/Mine a new block with new transactions.

  The transactions must be verified.
  """
  merkle_root = _FindMerkleRoot([_HashTransaction(transaction)[0]
                                 for transaction in transactions])
  block_header = plebcoin_pb2.BlockHeader(merkle_root=merkle_root,
                                          prev_hash=prev_hash,
                                          length=length)
  block = plebcoin_pb2.Block(header=block_header)
  block.transactions.extend(transactions)

  _MineBlock(block)

  return block


def _VerifyBlock(block, transactions, block_metadata):
  if not _BlockHashIsCorrect(_HashBlock(block)):
    return False
  if block.header.prev_hash not in block_metadata:
    return False
  last_metadata = block_metadata[block.header.prev_hash]
  if last_metadata.length + 1 != block.header.length:
    return False
  for transaction in block.transactions:
    if not _VerifyTransaction(transaction, transactions,
                              last_metadata.used_transactions):
      return False
  return True


BlockMetadata = collections.namedtuple('BlockMetadata', ['length',
                                                         'open_transactions',
                                                         'used_transactions'])

class Node(object):

  def __init__(self, public_key_string, in_queue, out_queues):
    self.blocks = {}
    self.block_metadata = {'0': BlockMetadata(0, 0, set(), set())}

    self.head = '0'
    self.head_metadata = BlockMetadata(0, 0, set(), set())

    self.transactions = {}
    self.used_transactions = set()

    self.public_key_string = public_key_string

    self.in_queue = in_queue
    self.out_queues = out_queues

  def Process(self, announce):
    if announce.sender == self.public_key_string:
      return
    if announce.HasField('transaction'):
      self._AcceptTransaction(announce.transaction)
    if announce.HasField('block'):
      self._AcceptBlock(announce.block)
    if announce.HasField('mine_empty_block'):
      self.AnnounceBlock()

  def AnnounceBlock(self):
    reward_transaction = plebcoin_pb2.Transaction(is_reward=True)
    reward_transaction.outputs.add(amount=50.0, receiver=self.public_key_string)
    reward_transaction.hash, _ = _ComputTransactionHash(reward_transaction)
    transactions = [self.transactions[thash]
                    for thash in self.head_metadata.open_transactions]
    new_block = _CreateBlock(
        transactions + [reward_transaction],
        self.head,
        self.head_metadata.length+1)
    self._AcceptBlock(new_block)
    self.Announce(plebcoin_pb2.Announce(block=new_block,
                                        sender=self.public_key_string))

  def _AcceptTransaction(self, transaction):
    transaction_hash, _ = _HashTransaction(transaction)
    if transaction_hash in self.transactions:
      return

    if not _VerifyTransaction(transaction, self.transactions,
                              self.head_metadata.used_transactions):
      return
    self.transactions[transaction_hash] = transaction
    self.head_metadata.open_transactions.add(transaction_hash)

    # Broadcast the transactions for others
    announce = plebcoin_pb2.Announce(transaction=transaction)
    self.Announce(announce)

    if len(self.head_metadata.open_transactions) == 1:
      self.AnnounceBlock()


  def _AcceptBlock(self, block):
    block_hash = _HashBlock(block)
    if block_hash in self.blocks:
      return
    if not _VerifyBlock(block, self.transactions, self.block_metadata):
      return
    self.blocks[block_hash] = block
    open_transactions = self.block_metadata[block.header.prev_hash].open_transactions.copy()
    used_transactions = self.block_metadata[block.header.prev_hash].used_transactions.copy()

    for transaction in block.transactions:
      transaction_hash , _ = _HashTransaction(transaction)
      self.transactions[transaction_hash] = transaction
      if transaction_hash in open_transactions:
        open_transactions.remove(transaction_hash)
      for tinput in transaction.inputs:
        used_transactions.add((tinput.sender, tinput.transaction))

    # We store time negatively to make comparing easier.
    new_metadata = BlockMetadata(length=block.header.length,
                                 open_transactions=open_transactions,
                                 used_transactions=used_transactions)
    self.block_metadata[block_hash] = new_metadata
    if new_metadata > self.head_metadata:
      self.head = block_hash
      self.head_metadata = new_metadata
    # Broadcast the block for others
    announce = plebcoin_pb2.Announce(block=block)
    self.Announce(announce)

KeyPair = collections.namedtuple('KeyPair', ['private_key', 'public_key',
                                             'public_key_string'])

def _GenerateKeyPair():
  private_key = dsa.generate_private_key(2048, backends.default_backend())
  public_key = private_key.public_key()
  public_key_bytes = public_key.public_bytes(
      serialization.Encoding.PEM,
      serialization.PublicFormat.SubjectPublicKeyInfo)
  public_key_string = binascii.hexlify(public_key_bytes)
  return KeyPair(private_key, public_key, public_key_string)



class Wallet(object):

  def __init__(self, in_queue, out_queues, response_queue, name):
    self.public_keys = set()
    self.private_keys = {}
    self.balance = {'0': 0}
    self.blocks = {}
    self.transactions = {'0': {}}
    self.block_metadata = {}

    self.head = '0'
    self.head_metadata = BlockMetadata(0, 0, set(), set())

    self.in_queue = in_queue
    self.out_queues = out_queues
    self.response_queue = response_queue

    self.name = name

  def Balance(self):
    return self.balance[self.head]

  def AddKeyPair(self, key_pair):
    self.public_keys.add(key_pair.public_key_string)
    self.private_keys[key_pair.public_key_string] = key_pair.private_key

  def Process(self, announce):
    if announce.HasField('block'):
      self._ProcessBlock(announce.block)
      return
    if announce.HasField('payment'):
      self.Send(announce.payment.address,
                announce.payment.amount)
    if announce.HasField('value_query'):
      self.response_queue.put((True, "", self.Balance()))
    if announce.HasField('get_public_key'):
      key_pair = _GenerateKeyPair()
      self.AddKeyPair(key_pair)
      self.response_queue.put((True, key_pair.public_key_string, self.Balance()))


  def _ProcessBlock(self, block):
    block_hash = _HashBlock(block)
    self.blocks[block_hash] = block

    # We store time negatively to make comparing easier.
    new_metadata = BlockMetadata(length=block.header.length,
                                 open_transactions=None,
                                 used_transactions=None)
    self.block_metadata[block_hash] = new_metadata
    if new_metadata > self.head_metadata:
      self.head = block_hash
      self.head_metadata = new_metadata

    transactions = self.transactions[block.header.prev_hash].copy()
    self.transactions[block_hash] = transactions
    self.balance[block_hash] = self.balance[block.header.prev_hash]

    for transaction in block.transactions:
      for tinput in transaction.inputs:
        if tinput.sender in self.public_keys:
          key = (tinput.sender, tinput.transaction)
          if key in transactions:
            self.balance[block_hash] -= transactions[key]
            del transactions[key]
      for output in transaction.outputs:
        if output.receiver in self.public_keys:
          transaction_hash, _ = _HashTransaction(transaction)
          key = (output.receiver, transaction_hash)
          if key not in transactions:
            transactions[key] = output.amount
            self.balance[block_hash] += output.amount

  def Send(self, address, amount):
    if amount > self.Balance():
      self.response_queue.put((False, "Can't send more than is in wallet", self.Balance()))
      return
    transactions_to_use = []
    pot = 0.0
    for key, value in self.transactions[self.head].iteritems():
      transactions_to_use.append(key)
      pot += value
      if pot > amount:
        break
    transaction = plebcoin_pb2.Transaction()
    change = pot - amount
    for sender, input_transaction in transactions_to_use:
      tinput = transaction.inputs.add()
      tinput.sender = sender
      tinput.transaction = input_transaction
    transaction.outputs.add(receiver=address, amount=amount)

    key_pair = _GenerateKeyPair()
    self.AddKeyPair(key_pair)

    transaction.outputs.add(receiver=key_pair.public_key_string,
                             amount=change)

    transaction.hash, hash_bytes = _HashTransaction(transaction)

    for sender, _ in transactions_to_use:
      private_key = self.private_keys[sender]
      signature_bytes = private_key.sign(hash_bytes,
                                         utils.Prehashed(hashes.SHA256()))
      signature = binascii.hexlify(signature_bytes)
      transaction.signatures.append(signature)

    self.Announce(plebcoin_pb2.Announce(transaction=transaction))
    self.response_queue.put((True, "Submitted transfer request", self.Balance()))


def main():
  logging.info("TODO")

if __name__ == '__main__':
  main()
