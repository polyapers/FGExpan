import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import math
    


class CGExpan(object):
    
    # Prerequisites
    hearst_patterns = [
        "t such as e",
        "such t as e",
        "e or other t",
        "e and other t",
        "t including e",
        "t, especially e"
    ]
    
    def instantiate_hearst_pattern(pattern, e, t):
        return pattern.replace("e", e).replace("t", t)
    
    class TypeGeneratingQuery:
        def __init__(self, pattern, entity):
            self.pattern = pattern
            self.entity = entity
    
        def generate_query(self):
            return instantiate_hearst_pattern(self.pattern, self.entity, "[MASK]")
    
    class EntityGeneratingQuery:
        def __init__(self, pattern, entity):
            self.pattern = pattern
            self.entity = entity
    
        def generate_query(self):
            return instantiate_hearst_pattern(self.pattern, "[MASK]", self.entity)
    
    def create_entailment_query(sentence, entity, entity_type):
        return f"In this sentence, {entity} is a type of {entity_type}."
    
    class PredictedTokens:
        def __init__(self, tokens):
            self.tokens = tokens
    
        def get_top_tokens(self, n):
            return self.tokens[:n]
    
    class MaskedTokenEmbedding:
        def __init__(self, embedding):
            self.embedding = embedding
    
    class LocalEmbedding:
        def __init__(self, embedding):
            self.embedding = embedding
    
    class GlobalEmbedding:
        def __init__(self, embedding):
            self.embedding = embedding
    
    class PretrainedEmbedding:
        def __init__(self, embedding):
            self.embedding = embedding
    
    entity_dictionary = set()
    
    def calculate_average_embedding(embeddings):
        if len(embeddings) == 0:
            return None
        total_embedding = sum(embeddings)
        return total_embedding / len(embeddings)


  # Type Generation Score

  def type_generation_score(entity_set, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
      token_count = 0
  
      for pattern in hearst_patterns:
          query = TypeGeneratingQuery(pattern, entity_set[0])  
          query_text = query.generate_query()
 
          predicted_tokens = language_model.predict_tokens(query_text)
  
          for token in predicted_tokens.get_top_tokens(5): 
              token_embedding = language_model.get_embedding(token)
              entity_type_embedding = language_model.get_embedding(entity_type)
              similarity = cosine_similarity([token_embedding], [entity_type_embedding])[0][0]
  
              total_similarity += similarity
              token_count += 1
  
      if token_count == 0:
          return 0.0  
  
      return total_similarity / (pattern_count * token_count)


  # Entity Generation Score

  def entity_generation_score(entity_set, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
      token_count = 0
      entity_count = len(entity_set)
  
      for pattern in hearst_patterns:
          for entity in entity_set:
              query = EntityGeneratingQuery(pattern, entity)
              query_text = query.generate_query()
  
              predicted_tokens = language_model.predict_tokens(query_text)
  
              for token in predicted_tokens.get_top_tokens(5):  
                  token_embedding = language_model.get_embedding(token)
                  entity_embedding = language_model.get_embedding(entity)
  
                  similarity = cosine_similarity([token_embedding], [entity_embedding])[0][0]
  
                  total_similarity += similarity
                  token_count += 1
  
      if token_count == 0:
          return 0.0  
  
      return total_similarity / (pattern_count * token_count * entity_count)


  # Entailment Score

  nli_model = pipeline("textual-entailment")

  def entailment_score(entity_set, entity_type, sentence_samples):
      total_score = 0.0
      entity_count = len(entity_set)
      sample_count = len(sentence_samples)
  
      for entity in entity_set:
          for sentence in sentence_samples[entity]:
              hypothesis = f"In this sentence, {entity} is a type of {entity_type}."
  
              entailment_result = nli_model(sentence, hypothesis)
  
              entailment_score = entailment_result[0]["score"]
  
              total_score += entailment_score
  
      if entity_count == 0 or sample_count == 0:
          return 0.0  
  
      return total_score / (sample_count * entity_count)

  
  # Type Selection Score
  
  def linear_projection(score, min_score, max_score):
      return (score - min_score) / (max_score - min_score)
  
  def type_selection_score(entity_set, entity_type, hearst_patterns, sentence_samples):
      t_g_score = type_generation_score(entity_set, entity_type, hearst_patterns, language_model)
      e_g_score = entity_generation_score(entity_set, entity_type, hearst_patterns, language_model)
      e_t_score = entailment_score(entity_set, entity_type, sentence_samples)
  
      min_t_g_score = min(0, t_g_score)  
      max_t_g_score = max(0, t_g_score)
  
      min_e_g_score = min(0, e_g_score)  
      max_e_g_score = max(0, e_g_score)
  
      min_e_t_score = 0  
      max_e_t_score = 1

      t_g_score_proj = linear_projection(t_g_score, min_t_g_score, max_t_g_score)
      e_g_score_proj = linear_projection(e_g_score, min_e_g_score, max_e_g_score)
      e_t_score_proj = linear_projection(e_t_score, min_e_t_score, max_e_t_score)
  
      final_score = f(t_g_score_proj) + f(e_g_score_proj) * f(e_t_score_proj)
  
      return final_score

  
  # Taxonomy-Guided Auxiliary Type Selection

  def entity_generation_score_between_types(type_1, type_2, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
  
      for pattern in hearst_patterns:
          query_type_1 = EntityGeneratingQuery(pattern, type_1)
          query_type_2 = EntityGeneratingQuery(pattern, type_2)
          query_text_type_1 = query_type_1.generate_query()
          query_text_type_2 = query_type_2.generate_query()
  
          predicted_tokens_type_1 = language_model.predict_tokens(query_text_type_1)
          predicted_tokens_type_2 = language_model.predict_tokens(query_text_type_2)
  
          for token_type_1 in predicted_tokens_type_1.get_top_tokens(5):  
              for token_type_2 in predicted_tokens_type_2.get_top_tokens(5):  
                  token_embedding_type_1 = language_model.get_embedding(token_type_1)
                  token_embedding_type_2 = language_model.get_embedding(token_type_2)
  
                  similarity = cosine_similarity([token_embedding_type_1], [token_embedding_type_2])[0][0]
  
                  total_similarity += similarity
  
      if pattern_count == 0:
          return 0.0  
  
      return total_similarity / pattern_count

  # Function to perform positive type selection
  def positive_type_selection(entity_set, predicted_type, hearst_patterns, language_model):
      top_ranked_generated_type = None
      max_seg_score = 0.0
  
      for generated_type in generated_types: 
          seg_score = entity_generation_score_between_types(predicted_type, generated_type, hearst_patterns, language_model)
          if seg_score > max_seg_score:
              max_seg_score = seg_score
              top_ranked_generated_type = generated_type

      entity_set_seg_score = entity_generation_score(entity_set, predicted_type, hearst_patterns, language_model)
  
      if max_seg_score > entity_set_seg_score:
          positive_type = top_ranked_generated_type
      else:
          positive_type = predicted_type
  
      return positive_type

  # Function to perform negative type selection
  def negative_type_selection(entity_set, predicted_type, taxonomy_types, hearst_patterns, language_model):
      negative_types = set()
  
      for generated_type in generated_types: 
          max_seg_score = 0.0
  
          for irrelevant_type in taxonomy_types:
              seg_score = entity_generation_score_between_types(generated_type, irrelevant_type, hearst_patterns, language_model)
              if seg_score > max_seg_score:
                  max_seg_score = seg_score
  
          entity_set_seg_score = entity_generation_score(entity_set, predicted_type, hearst_patterns, language_model)
  
          if max_seg_score > entity_set_seg_score:
              negative_types.add(generated_type)
  
      confident_negative_types = set()
      for negative_type in negative_types:
          avg_seg_score = 0.0
          for other_negative_type in negative_types:
              if negative_type != other_negative_type:
                  avg_seg_score += entity_generation_score_between_types(negative_type, other_negative_type, hearst_patterns, language_model)
          avg_seg_score /= len(negative_types) - 1  
  
          if avg_seg_score > entity_set_seg_score:
              confident_negative_types.add(negative_type)
  
      return confident_negative_types
  

  # Entity Dictionary Enrichment

  def entity_type_bi_directional_generation_score(entity, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
  
      for pattern in hearst_patterns:
          query_entity_type = TypeGeneratingQuery(pattern, entity)
          query_text_entity_type = query_entity_type.generate_query()
  
          predicted_types = language_model.predict_types(query_text_entity_type)
  
          for predicted_type in predicted_types:
              entity_type_embedding = language_model.get_embedding(entity_type)
              predicted_type_embedding = language_model.get_embedding(predicted_type)
  
              similarity = cosine_similarity([entity_type_embedding], [predicted_type_embedding])[0][0]
  
              total_similarity += similarity
  
      if pattern_count == 0:
          return 0.0  
  
      return total_similarity / pattern_count

  def entailment_score_between_entity_and_type(entity, entity_type, sentence_samples):
      total_score = 0.0
      sample_count = len(sentence_samples)
  
      for sentence in sentence_samples[entity]:
          hypothesis = f"In this sentence, {entity} is a type of {entity_type}."
          entailment_result = nli_model(sentence, hypothesis)
          entailment_score = entailment_result[0]["score"]
          total_score += entailment_score
  
      if sample_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_score / sample_count

  def entity_dictionary_enrichment(entity_set, inferred_type, candidate_mentions, language_model, enrichment_threshold):
      valid_mentions = []
      avg_seed_set_score = 0.0
      for entity in entity_set:
          b_g_score = entity_type_bi_directional_generation_score(entity, inferred_type, hearst_patterns, language_model)
          e_t_score = entailment_score_between_entity_and_type(entity, inferred_type, sentence_samples)
          avg_seed_set_score += f(b_g_score) + f(e_t_score)
      avg_seed_set_score /= len(entity_set)
  
      for candidate_mention in candidate_mentions:
          b_g_score = entity_type_bi_directional_generation_score(candidate_mention, inferred_type, hearst_patterns, language_model)
          e_t_score = entailment_score_between_entity_and_type(candidate_mention, inferred_type, sentence_samples)

          combined_score = f(b_g_score) + f(e_t_score)
  
          if combined_score > avg_seed_set_score:
              valid_mentions.append(candidate_mention)
  
      if len(valid_mentions) > enrichment_threshold:
          entity_dictionary.update(valid_mentions)
  
      return entity_dictionary


    # Fine-Grained Type-Guided Entity Ranking.

    def fine_grained_entity_ranking(entity, entity_type, hearst_patterns, sentence_samples, language_model, local_scores, global_scores):

        b_g_score = entity_type_bi_directional_generation_score(entity, entity_type, hearst_patterns, language_model)
        
        e_t_score = entailment_score_between_entity_and_type(entity, entity_type, sentence_samples)

        score_i_loc = local_scores.get(entity, 0.0) 
        score_i_glb = global_scores.get(entity, 0.0) 
        
        final_score = math.pow(b_g_score * e_t_score * math.pow(score_i_loc, 4) * math.pow(score_i_glb, 4), 1/4)
        
        return final_score

