#!/bin/bash
exec "$@" 2> >(grep -v 'table of contents is empty' >&2)
