"""Sandbox execution helpers.

This package provides a small, stable entrypoint that can be executed inside
isolated environments (Docker containers, Daytona Sandboxes, Modal Functions).

The goal is to keep the "runner" interface as simple as possible:

- input: JSON file describing a tool invocation
- output: JSON file containing the serialized ExecutionResult

See :mod:`ts_agents.sandbox.runner`.
"""
