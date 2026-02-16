from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        #TODO:
        # In this override impl we just need to add extra actions, we need to propagate attachment to the Choice since
        # in DeploymentTool they were propagated to the stage only as files. The main goal here is show pictures in chat
        # (DIAL Chat support special markdown to load pictures from DIAL bucket directly to the chat)
        # ---
        # 1. Call parent function `_execute` and get result
        message = await super()._execute(tool_call_params)
        # 2. If attachments are present then filter only "image/png" and "image/jpeg"
        if isinstance(message, Message) and message.custom_content and message.custom_content.attachments:
        # 3. Append then as content to choice in such format `f"\n\r![image]({attachment.url})\n\r")`
            for attachment in message.custom_content.attachments:
                if attachment.type in ["image/png", "image/jpeg"]:
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")
        # 4. After iteration through attachment if message content is absent add such instruction:
        #    'The image has been successfully generated according to request and shown to user!'
        #    Sometimes models are trying to add generated pictures as well to content (choice), with this instruction
        #    we are notifing LLLM that it was done (but anyway sometimes it will try to add file 😅)
            if not message.content:
                message.content = 'The image has been successfully generated according to request and shown to user!'
        # 5. Return Message with tool role, content, custom_content and tool_call_id
        return message

    @property
    def deployment_name(self) -> str:
        # TODO: provide deployment name for model that you have added to DIAL Core config (dall-e-3)
        return "dall-e-3"

    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        return "image_generation_tool"

    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return "Tool for generating images based on text description. It uses DALL-E 3 model to create images from prompts. The tool accepts a prompt describing the desired image and returns the generated image as an attachment. The prompt should be detailed and specific to achieve the best results. The tool can also handle optional parameters for customizing the image output, such as size and style. This tool is ideal for creating visual content, illustrations, or any scenario where a textual description needs to be transformed into an image."
    
    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema:
        #  - prompt is string, description: "Extensive description of the image that should be generated.", required
        #  - there are 3 optional parameters: https://platform.openai.com/docs/guides/image-generation?image-generation-model=dall-e-3#customize-image-output
        #  - Sample: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/dall-e?tabs=dalle-3#call-the-image-generation-api
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated."
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": [
                        "1024x1024",
                        "1024x1792",
                        "1792x1024"
                    ],
                    "default": "1024x1024"
                },
                "style": {
                    "type": "string",
                    "description": "The style of the generated image. Must be one of `vivid` or `natural`. \n- `vivid` causes the model to lean towards generating hyperrealistic and dramatic images. \n- `natural` causes the model to produce more natural, less realistic looking images.",
                    "enum": [
                        "natural",
                        "vivid"
                    ],
                    "default": "natural"
                },
                "quality": {
                    "type": "string",
                    "description": "The quality of the image that will be generated. ‘hd’ creates images with finer details and greater consistency across the image.",
                    "enum": [
                        "standard",
                        "hd"
                    ],
                    "default": "standard"
                }
            },
            "required": ["prompt"],
        }

