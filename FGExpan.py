import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import math
    


class CGExpan(object):

  // initialization 

  def 

  # Type Generation Score
  # Function to calculate the Type Generation Score
  def type_generation_score(entity_set, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
      token_count = 0
  
      for pattern in hearst_patterns:
          # Generate the Type-generating query
          query = TypeGeneratingQuery(pattern, entity_set[0])  # Assume entity_set is a list
          query_text = query.generate_query()
  
          # Get the predicted tokens for the query
          predicted_tokens = language_model.predict_tokens(query_text)
  
          for token in predicted_tokens.get_top_tokens(5):  # Consider top 5 predicted tokens
              # Calculate the cosine similarity between embeddings
              token_embedding = language_model.get_embedding(token)
              entity_type_embedding = language_model.get_embedding(entity_type)
              similarity = cosine_similarity([token_embedding], [entity_type_embedding])[0][0]
  
              total_similarity += similarity
              token_count += 1
  
      if token_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_similarity / (pattern_count * token_count)

  # Entity Generation Score
  # Function to calculate the Entity Generation Score
  def entity_generation_score(entity_set, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
      token_count = 0
      entity_count = len(entity_set)
  
      for pattern in hearst_patterns:
          for entity in entity_set:
              # Generate the Entity-generating query
              query = EntityGeneratingQuery(pattern, entity)
              query_text = query.generate_query()
  
              # Get the predicted tokens for the query
              predicted_tokens = language_model.predict_tokens(query_text)
  
              for token in predicted_tokens.get_top_tokens(5):  # Consider top 5 predicted tokens
                  # Calculate the cosine similarity between embeddings
                  token_embedding = language_model.get_embedding(token)
                  entity_embedding = language_model.get_embedding(entity)
  
                  similarity = cosine_similarity([token_embedding], [entity_embedding])[0][0]
  
                  total_similarity += similarity
                  token_count += 1
  
      if token_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_similarity / (pattern_count * token_count * entity_count)


  # Entailment Score
  
  # Initialize the NLI model for entailment
  nli_model = pipeline("textual-entailment")
  
  # Function to calculate the Entailment Score
  def entailment_score(entity_set, entity_type, sentence_samples):
      total_score = 0.0
      entity_count = len(entity_set)
      sample_count = len(sentence_samples)
  
      for entity in entity_set:
          for sentence in sentence_samples[entity]:
              # Create the entailment hypothesis
              hypothesis = f"In this sentence, {entity} is a type of {entity_type}."
  
              # Calculate the entailment score using the NLI model
              entailment_result = nli_model(sentence, hypothesis)
  
              # Extract the score (assuming 'score' is a key in the result)
              entailment_score = entailment_result[0]["score"]
  
              total_score += entailment_score
  
      if entity_count == 0 or sample_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_score / (sample_count * entity_count)

  
  # Type Selection Score
  
  # Function for linear projection
  def linear_projection(score, min_score, max_score):
      return (score - min_score) / (max_score - min_score)
  
  # Function to calculate the Type Selection Score
  def type_selection_score(entity_set, entity_type, hearst_patterns, sentence_samples):
      # Calculating each score  
      t_g_score = type_generation_score(entity_set, entity_type, hearst_patterns, language_model)
      e_g_score = entity_generation_score(entity_set, entity_type, hearst_patterns, language_model)
      e_t_score = entailment_score(entity_set, entity_type, sentence_samples)
  
      # Calculate min and max scores for linear projection
      min_t_g_score = min(0, t_g_score)  # Assuming S_TG values can be negative
      max_t_g_score = max(0, t_g_score)
  
      min_e_g_score = min(0, e_g_score)  # Assuming S_EG values can be negative
      max_e_g_score = max(0, e_g_score)
  
      min_e_t_score = 0  # S_E_T scores are typically between 0 and 1
      max_e_t_score = 1
  
      # Apply linear projection to each score
      t_g_score_proj = linear_projection(t_g_score, min_t_g_score, max_t_g_score)
      e_g_score_proj = linear_projection(e_g_score, min_e_g_score, max_e_g_score)
      e_t_score_proj = linear_projection(e_t_score, min_e_t_score, max_e_t_score)
  
      # Combine the projected scores using weights or any other desired function
      final_score = f(t_g_score_proj) + f(e_g_score_proj) * f(e_t_score_proj)
  
      return final_score

  
  # Taxonomy-Guided Auxiliary Type Selection

  # Function to calculate Entity Generation Score (\(S_E_G\)) between two types
  def entity_generation_score_between_types(type_1, type_2, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
  
      for pattern in hearst_patterns:
          # Generate Entity-generating queries for both types
          query_type_1 = EntityGeneratingQuery(pattern, type_1)
          query_type_2 = EntityGeneratingQuery(pattern, type_2)
          query_text_type_1 = query_type_1.generate_query()
          query_text_type_2 = query_type_2.generate_query()
  
          # Get the predicted tokens for the queries
          predicted_tokens_type_1 = language_model.predict_tokens(query_text_type_1)
          predicted_tokens_type_2 = language_model.predict_tokens(query_text_type_2)
  
          for token_type_1 in predicted_tokens_type_1.get_top_tokens(5):  # Consider top 5 predicted tokens
              for token_type_2 in predicted_tokens_type_2.get_top_tokens(5):  # Consider top 5 predicted tokens
                  # Calculate the cosine similarity between embeddings
                  token_embedding_type_1 = language_model.get_embedding(token_type_1)
                  token_embedding_type_2 = language_model.get_embedding(token_type_2)
  
                  similarity = cosine_similarity([token_embedding_type_1], [token_embedding_type_2])[0][0]
  
                  total_similarity += similarity
  
      if pattern_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_similarity / pattern_count

  # Function to perform positive type selection
  def positive_type_selection(entity_set, predicted_type, hearst_patterns, language_model):
      # Calculate Entity Generation Score (\(S_E_G\)) between predicted type and generated types
      top_ranked_generated_type = None
      max_seg_score = 0.0
  
      for generated_type in generated_types:  # Replace with your generated types
          seg_score = entity_generation_score_between_types(predicted_type, generated_type, hearst_patterns, language_model)
          if seg_score > max_seg_score:
              max_seg_score = seg_score
              top_ranked_generated_type = generated_type
  
      # Compare \(S_E_G\) between top-ranked generated type and predicted type with \(S_E_G\) between entity set and predicted type
      entity_set_seg_score = entity_generation_score(entity_set, predicted_type, hearst_patterns, language_model)
  
      if max_seg_score > entity_set_seg_score:
          positive_type = top_ranked_generated_type
      else:
          positive_type = predicted_type
  
      return positive_type

  # Function to perform negative type selection
  def negative_type_selection(entity_set, predicted_type, taxonomy_types, hearst_patterns, language_model):
      negative_types = set()
  
      # Iterate through generated types
      for generated_type in generated_types:  # Replace with your generated types
          max_seg_score = 0.0
  
          # Compare \(S_E_G\) between generated type and irrelevant taxonomy types
          for irrelevant_type in taxonomy_types:  # Replace with irrelevant taxonomy types
              seg_score = entity_generation_score_between_types(generated_type, irrelevant_type, hearst_patterns, language_model)
              if seg_score > max_seg_score:
                  max_seg_score = seg_score
  
          # Compare maximum \(S_E_G\) score with \(S_E_G\) between entity set and predicted type
          entity_set_seg_score = entity_generation_score(entity_set, predicted_type, hearst_patterns, language_model)
  
          # Check if the generated type is a negative type
          if max_seg_score > entity_set_seg_score:
              negative_types.add(generated_type)
  
      # Further merge confident negative types based on average \(S_E_G\)
      confident_negative_types = set()
      for negative_type in negative_types:
          avg_seg_score = 0.0
          for other_negative_type in negative_types:
              if negative_type != other_negative_type:
                  avg_seg_score += entity_generation_score_between_types(negative_type, other_negative_type, hearst_patterns, language_model)
          avg_seg_score /= len(negative_types) - 1  # Exclude the current type
  
          if avg_seg_score > entity_set_seg_score:
              confident_negative_types.add(negative_type)
  
      # Return the set of negative types
      return confident_negative_types
  

  # Entity Dictionary Enrichment

  # Function to calculate the Entity-Type Bi-directional Generation Score (\(S_B_G\))
  def entity_type_bi_directional_generation_score(entity, entity_type, hearst_patterns, language_model):
      total_similarity = 0.0
      pattern_count = len(hearst_patterns)
  
      for pattern in hearst_patterns:
          # Generate Type-generating queries for entity
          query_entity_type = TypeGeneratingQuery(pattern, entity)
          query_text_entity_type = query_entity_type.generate_query()
  
          # Get the predicted types for the query
          predicted_types = language_model.predict_types(query_text_entity_type)
  
          for predicted_type in predicted_types:
              # Calculate the cosine similarity between embeddings
              entity_type_embedding = language_model.get_embedding(entity_type)
              predicted_type_embedding = language_model.get_embedding(predicted_type)
  
              similarity = cosine_similarity([entity_type_embedding], [predicted_type_embedding])[0][0]
  
              total_similarity += similarity
  
      if pattern_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_similarity / pattern_count


  # Function to calculate the Entailment Score (\(S_E_T\)) between entity and type
  def entailment_score_between_entity_and_type(entity, entity_type, sentence_samples):
      total_score = 0.0
      sample_count = len(sentence_samples)
  
      for sentence in sentence_samples[entity]:
          # Create the entailment hypothesis
          hypothesis = f"In this sentence, {entity} is a type of {entity_type}."
  
          # Calculate the entailment score using the NLI model
          entailment_result = nli_model(sentence, hypothesis)
  
          # Extract the score (assuming 'score' is a key in the result)
          entailment_score = entailment_result[0]["score"]
  
          total_score += entailment_score
  
      if sample_count == 0:
          return 0.0  # Avoid division by zero
  
      return total_score / sample_count

  # Function to perform entity dictionary enrichment
  def entity_dictionary_enrichment(entity_set, inferred_type, candidate_mentions, language_model, enrichment_threshold):
      valid_mentions = []
  
      # Calculate the average Entity-Type Bi-directional Generation Score (\(S_B_G\)) + Entailment Score (\(S_E_T\)) for the seed set
      avg_seed_set_score = 0.0
      for entity in entity_set:
          b_g_score = entity_type_bi_directional_generation_score(entity, inferred_type, hearst_patterns, language_model)
          e_t_score = entailment_score_between_entity_and_type(entity, inferred_type, sentence_samples)
          avg_seed_set_score += f(b_g_score) + f(e_t_score)
      avg_seed_set_score /= len(entity_set)
  
      # Check candidate mentions for validity
      for candidate_mention in candidate_mentions:
          # Calculate the Entity-Type Bi-directional Generation Score (\(S_B_G\)) for the candidate mention
          b_g_score = entity_type_bi_directional_generation_score(candidate_mention, inferred_type, hearst_patterns, language_model)
  
          # Calculate the Entailment Score (\(S_E_T\)) for the candidate mention
          e_t_score = entailment_score_between_entity_and_type(candidate_mention, inferred_type, sentence_samples)
  
          # Calculate the combined score
          combined_score = f(b_g_score) + f(e_t_score)
  
          # Check if the candidate mention is valid based on the threshold
          if combined_score > avg_seed_set_score:
              valid_mentions.append(candidate_mention)
  
      # If count of valid mentions is above the threshold, add them to the entity dictionary
      if len(valid_mentions) > enrichment_threshold:
          entity_dictionary.update(valid_mentions)
  
      return entity_dictionary


    # Fine-Grained Type-Guided Entity Ranking.

    # Function to calculate the Fine-Grained Type-Guided Entity Ranking score (\(S_E_R\))
    def fine_grained_entity_ranking(entity, entity_type, hearst_patterns, sentence_samples, language_model, local_scores, global_scores):
        # Calculate the Entity-Type Bi-directional Generation Score (\(S_B_G\))
        b_g_score = entity_type_bi_directional_generation_score(entity, entity_type, hearst_patterns, language_model)
        
        # Calculate the Entailment Score (\(S_E_T\))
        e_t_score = entailment_score_between_entity_and_type(entity, entity_type, sentence_samples)
        
        # Calculate the final score (\(S_E_R\)) based on the formula
        score_i_loc = local_scores.get(entity, 0.0)  # Local score for the entity
        score_i_glb = global_scores.get(entity, 0.0)  # Global score for the entity
        
        final_score = math.pow(b_g_score * e_t_score * math.pow(score_i_loc, 4) * math.pow(score_i_glb, 4), 1/4)
        
        return final_score

