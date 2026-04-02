#!/bin/bash

for conf in configs/examples/*.yaml; do
    echo "=== Testing ${conf} ==="
    python main.py --conf "${conf}"
    echo ""
done

echo "All tests completed!"
