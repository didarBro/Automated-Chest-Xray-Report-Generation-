import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ViT_GPT2(nn.Module):
    def __init__(self, vit_model, gpt2_model, vit_output_dim):
        super(ViT_GPT2, self).__init__()
        self.vit = vit_model
        self.gpt2 = gpt2_model
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(len(tokenizer))
        self.image_to_embedding = nn.Linear(vit_output_dim, self.gpt2.config.n_embd)

    def forward(self, images, reports, attention_mask=None):
        device = next(self.parameters()).device  # Ensure device is defined
        images = images.to(device)
        reports = reports.to(device)
        
        image_features = self.vit(images)
        if image_features.dim() == 3:
            image_embeddings = self.image_to_embedding(image_features[:, 0, :])
        else:
            image_embeddings = self.image_to_embedding(image_features)
        
        image_embeddings = image_embeddings.unsqueeze(1)
        report_embeddings = self.gpt2.transformer.wte(reports)
        gpt2_inputs = torch.cat((image_embeddings, report_embeddings[:, :-1, :]), dim=1)
        outputs = self.gpt2(inputs_embeds=gpt2_inputs, labels=reports, attention_mask=attention_mask)
        return outputs

    def generate(self, images, device, max_length=256, num_beams=5, repetition_penalty=2.0, early_stopping=True):
        images = images.to(device)
        
        image_features = self.vit(images)
        if image_features.dim() == 3:
            image_embeddings = self.image_to_embedding(image_features[:, 0, :])
        else:
            image_embeddings = self.image_to_embedding(image_features)
            
        image_embeddings = image_embeddings.unsqueeze(1)

        # Generate with pad_token_id set to eos_token_id and using attention mask
        generated_ids = self.gpt2.generate(
            inputs_embeds=image_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
            pad_token_id=self.gpt2.config.eos_token_id,
            attention_mask=torch.ones(image_embeddings.shape[:2], device=device)
        )
        
        return generated_ids
