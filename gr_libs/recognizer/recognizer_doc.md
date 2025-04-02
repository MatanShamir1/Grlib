# Recognizer Module Documentation

This document provides an overview of the recognizer module, including its class hierarchy and instructions for adding a new class of recognizer.

## Class Hierarchy

The recognizer module consists of an abstract base class `Recognizer` and several derived classes, each implementing specific behaviors. The main classes are:

1. **Recognizer (Abstract Base Class)**
   - `inference_phase()` (abstract method)

2. **LearningRecognizer (Extends Recognizer)**
   - `domain_learning_phase()`

3. **GaAgentTrainerRecognizer (Extends Recognizer)**
   - `goals_adaptation_phase()` (abstract method)
   - `domain_learning_phase()`

4. **GaAdaptingRecognizer (Extends Recognizer)**
   - `goals_adaptation_phase()` (abstract method)

5. **GRAsRL (Extends Recognizer)**
   - Implements `goals_adaptation_phase()`
   - Implements `inference_phase()`

6. **Specific Implementations:**
   - `Graql (Extends GRAsRL, GaAgentTrainerRecognizer)`
   - `Draco (Extends GRAsRL, GaAgentTrainerRecognizer)`
   - `GCDraco (Extends GRAsRL, LearningRecognizer, GaAdaptingRecognizer)`
   - `Graml (Extends LearningRecognizer)`

## How to Add a New Recognizer Class

To add a new class of recognizer, follow these steps:

1. **Determine the Type of Recognizer:**
   - Will it require learning? Extend `LearningRecognizer`.
   - Will it adapt goals dynamically? Extend `GaAdaptingRecognizer`.
   - Will it train agents for new goals? Extend `GaAgentTrainerRecognizer`.
   - Will it involve RL-based recognition? Extend `GRAsRL`.

2. **Define the Class:**
   - Create a new class that extends the appropriate base class(es).
   - Implement the required abstract methods (`inference_phase()`, `goals_adaptation_phase()`, etc.).

3. **Initialize the Recognizer:**
   - Ensure proper initialization by calling `super().__init__(*args, **kwargs)`.
   - Set up any necessary agent storage or evaluation functions.

4. **Implement Core Methods:**
   - Define how the recognizer processes inference sequences.
   - Implement learning or goal adaptation logic if applicable.

5. **Register the Recognizer:**
   - Ensure it integrates properly with the existing system by using the correct `domain_to_env_property()`.

6. **Test the New Recognizer:**
   - Run experiments to validate its behavior.
   - Compare results against existing recognizers to ensure correctness.

By following these steps, you can seamlessly integrate a new recognizer into the framework while maintaining compatibility with the existing structure.
