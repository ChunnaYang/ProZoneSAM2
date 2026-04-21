#!/bin/bash

# Test the segmentation API endpoint with multiple boxes

echo "Testing Segmentation API with Multiple Boxes..."

# Test 1: Only WG box
echo -e "\n1. Testing Only WG Box..."
curl -X POST http://localhost:5000/api/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
    "boxes": [
      {"id": "box-1", "type": "WG", "x": 100, "y": 100, "width": 200, "height": 200}
    ],
    "useMedical": true
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | python3 -m json.tool | head -20

# Test 2: Only CG box
echo -e "\n2. Testing Only CG Box..."
curl -X POST http://localhost:5000/api/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
    "boxes": [
      {"id": "box-2", "type": "CG", "x": 150, "y": 150, "width": 100, "height": 100}
    ],
    "useMedical": true
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | python3 -m json.tool | head -20

# Test 3: Both WG and CG boxes (should return WG, CG, and PZ)
echo -e "\n3. Testing Both WG and CG Boxes (should return WG, CG, PZ)..."
curl -X POST http://localhost:5000/api/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
    "boxes": [
      {"id": "box-3", "type": "WG", "x": 100, "y": 100, "width": 200, "height": 200},
      {"id": "box-4", "type": "CG", "x": 150, "y": 150, "width": 100, "height": 100}
    ],
    "useMedical": true
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | python3 -m json.tool | head -30

echo -e "\n✅ API tests completed."
