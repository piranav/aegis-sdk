"""Aegis governance plugin for coding assistants.

One shared core (connection, liveness, evaluation, outage handling) and one small
adapter per assistant that translates its hook events and decision formats.
Standard library only, so it runs wherever the assistant runs hooks with python3.
"""

VERSION = "0.3.1"
