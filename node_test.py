"""Tests for google3.experimental.users.npat.hackathon.node."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import hashlib
import unittest

import cryptography
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import utils
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

class NodeTest(unittest.TestCase):

  def testCorrectTransactionIsVerified(self):
    private_key = dsa.generate_private_key(2048, backends.default_backend())
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    public_key_string = binascii.hexlify(public_key_bytes)

    first_transaction = plebcoin_pb2.Transaction()
    output = first_transaction.outputs.add()
    output.amount = 10.0
    output.receiver = public_key_string
    first_transaction_hash = node._HashTransaction(first_transaction)
    transactions = {first_transaction_hash: first_transaction}

    transaction = plebcoin_pb2.Transaction()
    transaction.inputs.add(sender=public_key_string,
                           transaction=first_transaction_hash)
    transaction.outputs.add(amount=10.0, receiver='1')

    transaction.hash, hash_bytes = (
        node._ComputeInnerTransactionHash(transaction))
    signature_bytes = private_key.sign(hash_bytes,
                                       utils.Prehashed(hashes.SHA256()))
    signature = binascii.hexlify(signature_bytes)
    transaction.signatures.append(signature)

    self.assertTrue(node._VerifyTransaction(transaction, transactions, set()))


class TestBlockChainMain(unittest.TestCase):

  def testGettingAndSettingAValue(self):
    pass


if __name__ == '__main__':
  unittest.main()
