# COCO Classes Reference

Complete list of 80 COCO object classes that YOLO can detect.

## All Classes (ID: Name)

### People & Animals (0-24)
- 0: person
- 14: bird
- 15: cat
- 16: dog
- 17: horse
- 18: sheep
- 19: cow
- 20: elephant
- 21: bear
- 22: zebra
- 23: giraffe

### Vehicles (1-8)
- 1: bicycle
- 2: car
- 3: motorcycle
- 4: airplane
- 5: bus
- 6: train
- 7: truck
- 8: boat

### Outdoor Objects (9-13)
- 9: traffic light
- 10: fire hydrant
- 11: stop sign
- 12: parking meter
- 13: bench

### Accessories (24-31)
- 24: backpack
- 25: umbrella
- 26: handbag
- 27: tie
- 28: suitcase

### Sports Equipment (29-43)
- 29: frisbee
- 30: skis
- 31: snowboard
- 32: sports ball
- 33: kite
- 34: baseball bat
- 35: baseball glove
- 36: skateboard
- 37: surfboard
- 38: tennis racket

### Kitchen & Dining (39-55)
- 39: bottle
- 40: wine glass
- 41: cup
- 42: fork
- 43: knife
- 44: spoon
- 45: bowl
- 46: banana
- 47: apple
- 48: sandwich
- 49: orange
- 50: broccoli
- 51: carrot
- 52: hot dog
- 53: pizza
- 54: donut
- 55: cake

### Furniture (56-61)
- 56: chair
- 57: couch (sofa)
- 58: potted plant
- 59: bed
- 60: dining table
- 61: toilet

### Electronics (62-65, 72-73)
- 62: tv (monitor)
- 63: laptop
- 64: mouse
- 65: remote
- 66: keyboard
- 67: cell phone
- 72: tv (television)
- 73: laptop

### Appliances (68-71)
- 68: microwave
- 69: oven
- 70: toaster
- 71: sink
- 72: refrigerator

### Indoor Objects (74-79)
- 74: book
- 75: clock
- 76: vase
- 77: scissors
- 78: teddy bear
- 79: hair drier
- 80: toothbrush

## Usage Examples

### Detect All Furniture
```bash
python main.py image.jpg --classes 56 57 58 59 60 61
```

### Detect People and Animals
```bash
python main.py image.jpg --classes 0 14 15 16 17 18 19 20 21 22 23
```

### Detect Electronics
```bash
python main.py image.jpg --classes 62 63 64 65 66 67
```

### Detect Kitchen Items
```bash
python main.py image.jpg --classes 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55
```

### Detect Vehicles
```bash
python main.py image.jpg --classes 1 2 3 4 5 6 7 8
```

## Common Room Segmentation Presets

### Living Room
```bash
# Furniture and common living room items
python main.py image.jpg --classes 56 57 58 60 62 63 75 76
# chair, couch, potted plant, dining table, tv, laptop, clock, vase
```

### Bedroom
```bash
# Bedroom furniture and items
python main.py image.jpg --classes 56 59 74 75 76
# chair, bed, book, clock, vase
```

### Kitchen
```bash
# Kitchen and dining items
python main.py image.jpg --classes 39 40 41 42 43 44 45 56 60 68 69 70 71 72
# bottle, wine glass, cup, fork, knife, spoon, bowl, chair, dining table, 
# microwave, oven, toaster, sink, refrigerator
```

### Office
```bash
# Office equipment and furniture
python main.py image.jpg --classes 56 60 62 63 64 65 66 67 74 75
# chair, dining table, tv, laptop, mouse, remote, keyboard, cell phone, book, clock
```

## Tips

1. **Multiple Classes**: You can combine any classes you want
   ```bash
   python main.py image.jpg --classes 0 56 57 60  # person, chair, couch, table
   ```

2. **Confidence Threshold**: Adjust to get more or fewer detections
   ```bash
   python main.py image.jpg --conf 0.5  # Higher = fewer but more confident
   ```

3. **Model Size**: Larger models are more accurate but slower
   - `yolov8n.pt` - Fastest, good for testing
   - `yolov8s.pt` - Small, balanced
   - `yolov8m.pt` - Medium, recommended for production
   - `yolov8l.pt` - Large, high accuracy
   - `yolov8x.pt` - Extra large, best accuracy

4. **Find Class IDs**: If unsure about a class ID, run without `--classes` first to see what objects are detected.
