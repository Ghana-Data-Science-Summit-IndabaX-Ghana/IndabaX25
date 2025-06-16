import spacy
import random
from pathlib import Path
from spacy.tokens import DocBin
from spacy.training.example import Example # Correct import for Example
from spacy.util import minibatch
from tqdm import tqdm
import re 
from nervaluate import Evaluator

# anything that is **not** a letter or digit → replace
chars_to_ignore_regex = r"[^A-Za-z0-9]+"

def remove_special_characters(text):
    """
    Replace non-alphanumeric runs with a single space and lowercase the result.
    """
    cleaned = re.sub(chars_to_ignore_regex, " ", text)   # keep letters & digits
    cleaned = re.sub(r"\s+", " ", cleaned)               # squeeze repeated spaces
    return cleaned.lower().strip()

class BioNerDataProcessor:
    """
    Class to handle reading and processing of .bio (CoNLL-style) NER data.
    """
    def __init__(self, bio_file_path):
        self.bio_file_path = Path(bio_file_path)
        if not self.bio_file_path.is_file():
            raise FileNotFoundError(f"BIO file not found at: {self.bio_file_path}")


    def read_bio_file(self):
      """
      Reads a CoNLL-style .bio file and returns a list of
      (text, {"entities": [(start, end, label), …]}) tuples.
      """
      TRAIN_DATA , entities =[], []
      entity_start_char_time, entity_end_char_time = 0,0
      sentence ,tag_idx="", ""
      stored_entities={}

      with open(self.bio_file_path, mode='r',encoding="utf8") as f:
        lines = f.readlines()
      
      
      for num,line in enumerate(lines):
        line= line.strip()
        
        part =line.split()
        if len(part)==2:
        
          token, tag = line.split()
          sentence = sentence + tag + " "
        
          if tag:
            entity_end_char_time= entity_start_char_time + len(tag.strip())
            if token.startswith("O"):
              entity_start_char_time=entity_end_char_time+1
              continue
            elif token.split("-"):
              idx= token.split("-")[0]
    
              if token.split("-")[1] in stored_entities:
                stored_entities[token.split("-")[1]]["end"]=entity_end_char_time
                if idx=="B":
                    stored_entities[token.split("-")[1]]["start"]=entity_start_char_time
      
              else:
                  stored_entities[token.split("-")[1]]={"start":entity_start_char_time,"end":entity_end_char_time}
            
              entity_start_char_time=entity_end_char_time+1
        if not line: #this means if it reads an empty line
          entity_start_char_time, entity_end_char_time = 0,0
          
          
          entities= [(values[1]['start'],values[1]['end'],values[0]) for values in stored_entities.items()] 

          #we normalize the text before passing it 
          sentence = remove_special_characters(sentence)
          TRAIN_DATA.append((sentence.strip(),{"entities":entities}))
          sentence ,tag_idx="", ""

          stored_entities={}

      return TRAIN_DATA

    def preprocess_data(self, raw_data):
        """
        Placeholder for any further filtering or augmentation.
        For now, just returns it unchanged.
        Can be extended to, e.g., filter out examples with no entities.
        """
        # Example: Filter out sentences with no entities
        processed_data = [(text, annots) for text, annots in raw_data if annots.get("entities")]
        return processed_data
        


class SpacyNerTrainer:
    """
    Class to handle training a spaCy NER model.
    """
    def __init__(self, blank_model_name="en", n_iterations=20, dropout_rate=0.2, batch_size=128, output_dir="./custom_ner_model"):
        self.blank_model_name = blank_model_name
        self.n_iterations = n_iterations
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.nlp = spacy.blank(self.blank_model_name)
        self._setup_ner_pipe()

    def _setup_ner_pipe(self):
        """Initializes or gets the NER pipe."""
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")

    def _add_labels_to_ner(self, train_data):
        """Adds all unique entity labels from training data to the NER component."""
        all_labels = set()
        for _, annotations in train_data:
            for ent_start, ent_end, label in annotations.get("entities", []):
                all_labels.add(label)
        for label in all_labels:
            self.ner.add_label(label)
        print(f"Added labels to NER: {sorted(list(all_labels))}")

    def train_model(self, train_data):
        """Trains the NER model."""
        self._add_labels_to_ner(train_data)

        # Prepare example data for initialization (can use a subset)
        # Ensure TRAIN_DATA is not empty
        if not train_data:
            print("Error: TRAIN_DATA is empty. Cannot initialize model.")
            return

        example_data_for_init = [
            Example.from_dict(self.nlp.make_doc(text), annots)
            for text, annots in train_data[:min(100, len(train_data))] # Use a subset or all
        ]
        if not example_data_for_init:
            print("Error: Could not create example data for initialization from TRAIN_DATA.")
            return


        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = self.nlp.initialize(get_examples=lambda: example_data_for_init)

            for itn in range(self.n_iterations):
                random.shuffle(train_data)
                epoch_losses = {}
                batches = minibatch(train_data, size=self.batch_size)

                for batch in tqdm(batches, desc=f"Epoch {itn + 1}/{self.n_iterations}"):
                    examples_for_update = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        examples_for_update.append(Example.from_dict(doc, annotations))

                    if examples_for_update:
                        self.nlp.update(
                            examples_for_update,
                            drop=self.dropout_rate,
                            sgd=optimizer,
                            losses=epoch_losses
                        )
                if 'ner' in epoch_losses and examples_for_update: # Check if examples_for_update is not empty
                    avg_loss = epoch_losses['ner'] / len(examples_for_update) # More accurate avg per batch step in last batch
                    print(f"Epoch {itn + 1} finished. Average NER Loss (last batch): {avg_loss:.4f} (Total Epoch: {epoch_losses['ner']:.2f})")
                else:
                    print(f"Epoch {itn + 1} finished. Losses: {epoch_losses}")


    def save_model(self):
        """Saves the trained model to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(self.output_dir)
        print(f"Saved model to {self.output_dir}")

    def test_model(self, test_text):
        """Loads the saved model and tests it on a sample text."""
        print("\n--- Loading and Testing Model ---")
        try:
            nlp_loaded = spacy.load(self.output_dir)
            doc = nlp_loaded(test_text)
            print(f"Test text: '{test_text}'")
            print("Entities found:", [(ent.text, ent.label_) for ent in doc.ents])
        except Exception as e:
            print(f"Error loading or testing model: {e}")
    def make_dicts_from_offsets(self,text, spans):
      """
      Convert a span list to the dictionaries nervaluate expects.

      If pred == False → spans = (start, end, label)
      If pred == True  → spans = (entity_text, label, start, end)
      """
      out = []
 
      for span in spans:
     
          start, end, label = span
          ent_text = text[start:end]
          out.append({
              "text": ent_text,
              "label": label,
              "start": start,
              "end": end,
          })
      return out


def main():
    """Main function to run the NER training pipeline."""
    # --- 1. CONFIGURABLE PARAMETERS ---
    n_iterations = 10
    blank_model_name = "en"
    bio_file_path = "restauranttrain.bio" # Make sure this file exists or change path
    output_model_dir = "./restaurant_ner_model_v2"
    dropout = 0.2
    batch_size_train = 32 # Adjusted batch size


    # --- 2. LOAD & PREPROCESS DATA ---
    print("Loading and preprocessing data...")
    data_processor = BioNerDataProcessor(bio_file_path=bio_file_path)
    raw_train_data = data_processor.read_bio_file()
    print(f"Loaded {len(raw_train_data)} annotated sentences. {raw_train_data[0]}")
    if not raw_train_data:
        print("No data loaded from BIO file. Exiting.")
        return

    train_data_full = data_processor.preprocess_data(raw_train_data)
    random.shuffle(train_data_full)

    # Split data (e.g., 80% train, 20% dev/test)
    split_ratio = 0.80
    split_index = int(len(train_data_full) * split_ratio)

    TRAIN_DATA = train_data_full[:split_index]
    DEV_DATA = train_data_full[split_index:] # Using DEV_DATA for evaluation later (optional)

    if not TRAIN_DATA:
        print("Not enough data for training after split. Exiting.")
        return

    print(f"Total examples: {len(train_data_full)}, Training examples: {len(TRAIN_DATA)}, Dev examples: {len(DEV_DATA)}")

    # --- 3. TRAIN MODEL ---
    print("\nSetting up and training NER model...")
    trainer = SpacyNerTrainer(
        blank_model_name=blank_model_name,
        n_iterations=n_iterations,
        dropout_rate=dropout,
        batch_size=batch_size_train,
        output_dir=output_model_dir
    )
    trainer.train_model(TRAIN_DATA)

    # --- 4. SAVE MODEL ---
    trainer.save_model()

    # --- 5. TEST MODEL ---
    test_text_example = "I want a cheap italian restaurant nearby for dinner."
    trainer.test_model(test_text_example)
  
    if DEV_DATA:
        print("\n--- Evaluating on Dev Data (micro-averaged) ---")

        # 1) load model and get label set
        nlp_loaded = spacy.load(output_model_dir)
        tags = sorted(nlp_loaded.pipe_labels["ner"])


        # 2) accumulate references / predictions sentence-by-sentence
        all_refs, all_preds = [], []
        with open ("dev_outputs.txt",'w') as f:
        
          for idx, sample in enumerate(DEV_DATA):
              text,gold_ann= sample
           
              doc = nlp_loaded(text)

              gold_spans = gold_ann["entities"]                         # (start,end,label)
              pred_spans = [(ent.start_char, ent.end_char, ent.label_)  # same shape
                            for ent in doc.ents]
              f.write(f"{idx+1}. Text: {text}\n")
              f.write(f"Gold Entities: {gold_spans}\n")
              f.write(f"Pred Entities: {pred_spans}\n")
              f.write('\n')
              # trainer.make_dicts_from_offsets → ♥ keeps your helper
              all_refs.append(trainer.make_dicts_from_offsets(text, gold_spans))
              all_preds.append(trainer.make_dicts_from_offsets(text, pred_spans))

          # 4) one evaluation pass, micro-averaged over all sentences
          evaluator = Evaluator(all_refs, all_preds, tags=tags, loader="default")
          overall, per_type, _, _ = evaluator.evaluate()   # nervaluate ≥0.2.x
          print("Overall:", overall)
          print("Per-type:", per_type)

          prec = overall["strict"]["precision"]
          rec  = overall["strict"]["recall"]
          f1   = overall["strict"]["f1"]

          print(f"Micro Precision: {prec:.2%}")
          print(f"Micro Recall:    {rec:.2%}")
          print(f"Micro F1:        {f1:.2%}")




if __name__ == "__main__":
    main()
