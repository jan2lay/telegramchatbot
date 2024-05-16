
import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()

#Telgram console log run view
logging.basicConfig(
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  level=logging.INFO)

#Start Telegram function
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
  docs = DATABASE.similarity_search(update.message.text, k=4)
  chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
  results = chain({
    'input_documents': docs,
    "question": update.message.text
  },
                  return_only_outputs=True)
  text = results['output_text']
  await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("query", cmd_query))
    

    # on non command i.e message - echo the message on Telegram
    #application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
