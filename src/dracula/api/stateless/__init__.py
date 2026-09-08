"""Contain deterministic stateless replay and its HTTP application.

Each request carries the seed and accepted history needed to reconstruct the
game without persistent server state.
"""
