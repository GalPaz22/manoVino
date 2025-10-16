require('dotenv').config();

/******************************************************
 * SERVER SCRIPT – Fetch from Mood + GPT Category/Embedding
 ******************************************************/
const express = require('express');
const axios = require('axios');

const { parse } = require('node-html-parser');
const fs = require('fs');
const path = require('path');
const { MongoClient } = require('mongodb');
const { OpenAIEmbeddings } = require('@langchain/openai');
const cron = require('node-cron');


// ========== 0) Basic Express Setup ==========
const app = express();
app.use(express.json());
const PORT = Number(process.env.PORT) || 3030; // allow override via env

// ========== 1) MongoDB Configuration ==========
const mongoUri = process.env.MONGO_URI;
const dbName = 'manoVino';
const collectionName = 'products';

// ========== 2) External API Configuration ==========
const externalApiUrl = process.env.EXTERNAL_API_URL;
const externalApiToken = process.env.EXTERNAL_API_TOKEN;

// ========== 3) OpenAI Setup ==========
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-large',
  apiKey: OPENAI_API_KEY,
});
const { OpenAI } = require('openai');
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// ========== 4) Helper Functions ==========

/** Strips HTML tags to get plain text. */
function parseHtmlToPlainText(html) {
  const root = parse(html || '');
  return root.textContent.trim();
}

/** Translates Hebrew (or any language) to English via GPT. */
async function translateDescription(description) {
  if (!description) return null;
  try {
    const translationResponse = await openai.chat.completions.create({
      model: 'gpt-4o-mini', // or 'gpt-3.5-turbo' if you prefer
      messages: [
        {
          role: 'user',
          content: `Translate the following text to English. answer only with the translated text:\n\n${description}`,
        },
      ],
    });
    const translatedText = translationResponse.choices?.[0]?.message?.content?.trim() || '';
    return translatedText;
  } catch (error) {
    console.warn('Translation failed:', error);
    return null;
  }
}

/** Generates a vector embedding for a text (already in English). */
async function generateEmbeddings(translatedText) {
  try {
    return await embeddings.embedQuery(translatedText);
  } catch (error) {
    console.warn('Embedding generation failed:', error);
    return null;
  }
}

/**
 * GPT-based category classification
 * e.g.: returns "יין אדום", or "סאקה", or "None"
 */
async function classifyCategoryUsingGPT(translatedDescription, productName) {
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o', // or 'gpt-3.5-turbo'
      messages: [
        {
          role: 'user',
          content: `Based on the following description, determine which category it belongs to from the list:
"יין לבן, יין אדום, יין רוזה, יין מבעבע, ליקר, וויסקי, וודקה, טקילה, מזקל, קוניאק, בירה, קוקטייל, רום, גין, סאקה, אפריטיף, דג׳יסטיף, ביטר, מארז".
Answer ONLY with the category name, nothing else! If none fit, return "None" (exactly).

Description: ${translatedDescription}
Product name: ${productName}`,
        },
      ],
    });

    let category = response.choices?.[0]?.message?.content?.trim() || '';
    if (category === 'None') {
      category = null;
    }
    return category;
  } catch (error) {
    console.warn('Category classification failed:', error);
    return null;
  }
}

// ========== 5) Mongo Client init ==========
const client = new MongoClient(mongoUri, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// ========== 6) Main fetchAndStore function ==========
/**
 * 1) Fetch from external Mood API
 * 2) Upsert items into Mongo
 * 3) Mark old items as notInStore
 * 4) Then run GPT-based category & embedding process on items that need it
 */
const fetchAndStoreData = async () => {
  try {
    console.log('📡 Initiating GET request to external API...');

    // 1) Fetch
    const response = await axios.get(externalApiUrl, {
      headers: { Authorization: externalApiToken },
      timeout: 30000,
    });

    console.log(`✅ Received response with status: ${response.status}`);

    const { Items, Success, Message } = response.data;
    if (!Success) {
      console.warn(`⚠️ API call failed with message: ${Message}`);
      throw new Error(`API call failed: ${Message}`);
    }
    if (!Array.isArray(Items)) {
      throw new Error('Unexpected response format: "Items" is not an array.');
    }
    console.log(`📦 Received ${Items.length} items from API.`);

    // 2) Upsert
    const db = client.db(dbName);
    const collection = db.collection(collectionName);
    const fetchedIds = [];

    // Pre-count how many active items were previously out of stock
    const activeIds = Items
      .filter(it => it && it.IsActive === true && it.ItemID)
      .map(it => it.ItemID);
    let previouslyOutToInCount = 0;
    if (activeIds.length > 0) {
      previouslyOutToInCount = await collection.countDocuments({
        id: { $in: activeIds },
        stockStatus: 'outofstock',
      });
    }

    // For your "wine type" mapping logic
    const wineTypeMap = {
      'לבן': 'יין לבן',
      'אדום': 'יין אדום',
      'רוזה': 'יין רוזה',
      'כתום': 'יין כתום',
      'מבעבע': 'יין מבעבע',
    };

    // For subcategory regex
    const categoriesWanted = [
      'סאקה',
      'בירות',
      'ברנדי',
      'קוניאק',
      'וויסקי',
      'אפירטיף',
      'דג׳יסטיף',
      'קוקטייל',
      'מארז',
      'ליקר', 
      'סיידר',
      'ארגז',
      'מזקל',
      'מארז',
    ];
    const categoryRegex = new RegExp(`(${categoriesWanted.join('|')})`, 'g');

    for (const item of Items) {
      const itemId = item.ItemID;
      if (!itemId) {
        console.warn('Item without an ItemID, skipping:', item);
        continue;
      }
      fetchedIds.push(itemId);
      item.name = item.Name;
      delete item.Name;

      item.seo= item.SEOData;
    
      item.url = item.ItemUrl;
      delete item.ItemUrl;

      item.price = item.Price;
      delete item.Price;
    
      // 2) Convert ItemGallery -> item.image
      //    (if you only want a single image, pick the first GalleryPhoto)
      if (
        item.ItemGallery &&
        Array.isArray(item.ItemGallery.GalleryPhotos) &&
        item.ItemGallery.GalleryPhotos.length > 0
      ) {
        item.image = item.ItemGallery.GalleryPhotos[0].FileUrl;
      } else {
        item.image = null;
      }
    
      // Clean description HTML
      const baseHtml = item.Description || '';
      const baseText = parse(baseHtml).textContent.trim();

      // stockStatus
      item.stockStatus = item.IsActive === true ? 'instock' : 'outofstock';

// Add this somewhere in your processing loop
// This is a more robust implementation with debugging

// First, let's log what we're working with
console.log(`Processing offers for item ${item.name || item.Name}, Offers:`, 
  item.Offers ? `Found ${item.Offers.length} offers` : 'No offers array');

// Check if there's at least one offer to examine field structure
if (item.Offers && item.Offers.length > 0) {
  console.log('Sample offer structure:', JSON.stringify(item.Offers[0], null, 2));
}

// Initialize specialSales array
// ...existing code...
if (Array.isArray(item.Offers)) {
  item.specialSales = item.Offers
    .filter(offer => {
      if (!offer) return false;
      
      const isActive = offer.Active === true || offer.active === true;
      const startDate = offer.StartDate || offer.startDate;
      const endDate = offer.EndDate || offer.endDate;
      let datesValid = true;
      
      if (startDate && endDate) {
        try {
          const start = new Date(startDate);
          const end = new Date(endDate);
          const now = new Date();
          datesValid = start <= now && end >= now;
        } catch (e) {
          console.warn('Date parsing error for offer:', e.message);
          datesValid = false;
        }
      }
      
      return isActive && datesValid;
    })
    .map(offer => {
      const fileUrl = (offer.LabelImage &&
                      typeof offer.LabelImage === 'object' &&
                      offer.LabelImage.FileUrl)
        ? offer.LabelImage.FileUrl
        : null;
    /*  if (!item.type.includes("מבצע")) {
        item.type.push("מבצע");
        item.onSale = true;
      }*/
      return {
        name: offer.Name || offer.name || null,
        fileUrl: fileUrl
      };
    })
    .filter(offer => offer.fileUrl !== null);
}
  
// Log the final result
console.log(`Final specialSales array for ${item.name || item.Name}:`, 
  item.specialSales.length > 0 ? item.specialSales : 'Empty array');

// CustomFields
let customFieldsExtra = '';
if (Array.isArray(item.CustomFields)) {
  item.CustomFields.forEach((field) => {
    if (field.Name && field.ItemValue && field.ItemValue.trim() !== '') {
      let finalValue = field.ItemValue.trim();
      if (finalValue === '1') {
        finalValue = 'ממליץ';
      }
      customFieldsExtra += `\n${field.Name}: ${finalValue}`;
    }
  });
}

      // --- Collections & ExtraCategories processing ---
      // We'll use two arrays:
      // - item.type: for flags like "כשר", "מבצע", etc.
      // - item.category: for local categories like "יין מבעבע" or from subcategory matching.
      // Reinitialize item.type so we start fresh.
      item.type = [];
      // Temporary flag to determine if we have a מבצע indication:
      let hasSale = false;
      // Temporary array to hold category values from Collections:
      let categoriesFromCollections = [];

      let groupMap = {};
      if (Array.isArray(item.Collections)) {
        groupMap = item.Collections.reduce((acc, col) => {
          if (col.GroupName && col.Name) {
            if (!acc[col.GroupName]) acc[col.GroupName] = [];
            acc[col.GroupName].push(col.Name);
          }
          // If "כשרות" == "כשר"
          if (col.GroupName === "כשרות" && col.Name === "כשר") {
            item.type.push("כשר");
          }
          // If "מבצעים"
          if (col.GroupName === "מבצעים") {
            hasSale = true;
            item.type.push("מבצע");
            item.onSale = true;
          }

          if (col.GroupName === "מידת יובש") {
            item.type.push(col.Name);
          }
          // If either GroupName or Name contains "מבעבע", push "יין מבעבע"
          if (
            (col.GroupName && col.GroupName.includes("מבעבע")) ||
            (col.Name && col.Name.includes("מבעבע"))
          ) {
            categoriesFromCollections.push("יין מבעבע");
          }
          return acc;
        }, {});

        // Check ExtraCategories for "מבצע"
       /* if (Array.isArray(item.ExtraCategories)) {
          for (const extraCat of item.ExtraCategories) {
            if (extraCat && extraCat.Name && extraCat.Name.includes("מבצע")) {
              hasSale = true;
              if (!item.type.includes("מבצע")) {
                item.type.push("מבצע");
              }
              item.onSale = true;
              break; // Once found, break out of the loop
            }
          }
        }*/
      }

      // If no sale indication was found, ensure "מבצע" is not present and onSale is false.
      if (!hasSale) {
        item.type = item.type.filter(x => x !== "מבצע");
        item.onSale = false;
      }

      // Convert groupMap -> appended text
      let collectionsExtra = '';
      for (const groupName of Object.keys(groupMap)) {
        const allNames = groupMap[groupName].join(', ');
        collectionsExtra += `\n${groupName}: ${allNames}`;
      }

      // Combine text
      const combinedText = baseText + customFieldsExtra + collectionsExtra;
      const finalDescription = parse(combinedText).textContent.trim();
      item.description = finalDescription;

      // --- Build local productCategory ---
      let productCategory = [];
      if (groupMap['סוג יין']) {
        const firstWineType = groupMap['סוג יין'][0];
        const wineType = wineTypeMap[firstWineType] || 'יין';
        productCategory.push(wineType, "יין");
      }
      if (item.Category && item.Category.SubCategory && item.Category.SubCategory.Name) {
        const subCatName = item.Category.SubCategory.Name;
        const matches = subCatName.match(categoryRegex);
        if (matches && matches.length > 0) {
          const uniqueMatches = [...new Set(matches)];
          productCategory = uniqueMatches;
        }
      }
// --- Merge local productCategory with categories from Collections ---
const mergedCategories = [...new Set([...productCategory, ...categoriesFromCollections])];

// Add "יין" to any wine-related categories
const wineCategories = ['יין אדום', 'יין לבן', 'יין מבעבע', 'יין רוזה', 'יין כתום'];
const hasWineCategory = mergedCategories.some(cat => wineCategories.includes(cat));
if (hasWineCategory && !mergedCategories.includes('יין')) {
  mergedCategories.push('יין');
}

// If there are any categories, assign them; otherwise, leave item.category unchanged.
if (mergedCategories.length > 0) {
  item.category = mergedCategories;
}

           //Force onSale true if price is 42, 45, 59, or 85
           if ([42, 45, 59, 85].includes(Number(item.price))) {
            item.onSale = true;
            // Add this line to ensure מבצע is in the type array
            if (!item.type.includes("מבצע")) {
              item.type.push("מבצע");
            }
          }

      // Upsert item in MongoDB
      await collection.updateOne(
        { id: itemId },
        {
          $set: {
            ...item,
            id: itemId,
            notInStore: false,
            updatedAt: new Date(),
          },
        },
        { upsert: true }
      );
    }

    console.log(`🔄 Restocked ${previouslyOutToInCount} items (were outofstock, now instock).`);

    // 3) Mark old items as notInStore
    if (fetchedIds.length > 0) {
      const result = await collection.updateMany(
        { id: { $nin: fetchedIds } },
        { $set: { notInStore: true } }
      );
      console.log(`🗑 Marked ${result.modifiedCount} old items as notInStore.`);
    }

    // 4) Now run GPT translation/embedding on items missing GPT-based category or embedding
    await processDescriptionsAndCategories();

    console.log('✅ Upsert complete + GPT category done.');
  } catch (error) {
    if (error.response) {
      console.error('❌ API Error:', {
        status: error.response.status,
        data: error.response.data,
      });
    } else if (error.request) {
      console.error('❌ No response received from external API:', error.request);
    } else {
      console.error('❌ Error in API request setup:', error.message);
    }
  }
};

/**
 * This function finds items that do NOT have:
 *   - embedding
 *   - or GPT-based category
 * and then:
 *   1) Translates their description to English
 *   2) Embeds the translation
 *   3) Classifies via GPT
 *   4) Stores them in the DB
 */
async function processDescriptionsAndCategories() {
  const db = client.db(dbName);
  const collection = db.collection(collectionName);

  // Find products missing "embedding" or "category"
  const products = await collection.find({
    $or: [
      { embedding: { $exists: false } },
      { category: { $exists: false } },
      { category: null },
      { category: [] }
    ],
    stockStatus: "instock"
  }).toArray();

  console.log(`Processing ${products.length} products for categories/embeddings`);

  for (const product of products) {
    try {
      const { _id, name, description, category, embedding } = product;
      if (!description) {
        console.log(`Skipping product ${_id} - no description`);
        continue;
      }

      // 1) Translate once (for GPT classification + embeddings)
      const translatedDescription = await translateDescription(description);
      if (!translatedDescription) {
        console.log(`Skipping product ${_id} - translation failed`);
        continue;
      }

      // Build fields to update
      const updateFields = {};

      // 2) If no GPT-based category, do GPT classification
      if (!category || (Array.isArray(category) && category.length === 0)) {
        const gptCategory = await classifyCategoryUsingGPT(translatedDescription, name);
        if (gptCategory) {
          const categories = [gptCategory];
          
          // Add "יין" to any wine-related categories
          const wineCategories = ['יין אדום', 'יין לבן', 'יין מבעבע', 'יין רוזה', 'יין כתום'];
          if (wineCategories.includes(gptCategory) && !categories.includes('יין')) {
            categories.push('יין');
          }
          
          updateFields.category = categories;
        }
        console.log(`Classified product ${_id} as category: ${gptCategory}`);
      }

      // 3) If no embedding, generate it
      if (!embedding) {
        const embeddedText = await generateEmbeddings(translatedDescription);
        if (embeddedText) {
          updateFields.embedding = embeddedText;
          console.log(`Generated embedding for product ${_id}`);
        }
      }

      // 4) Store the English translation
      updateFields.description1 = translatedDescription;

      if (Object.keys(updateFields).length > 0) {
        const result = await collection.updateOne(
          { _id }, 
          { $set: updateFields }
        );
        console.log(`Updated product "${name}" (id: ${_id}):`, updateFields, 'Result:', result.modifiedCount);
      }
    } catch (err) {
      console.error(`Error processing product "${product.name}":`, err);
    }
  }

  console.log('✅ Finished GPT-based processing on products.');
}

// ========== 7) Express Routes ==========
app.get('/fetch-mood-items', async (req, res) => {
  try {
    await fetchAndStoreData();
    res.status(200).json({ message: 'Data fetched, upserted, GPT processed.' });
  } catch (error) {
    console.error('❌ Error during manual fetch:', error.message);
    res.status(500).json({ error: 'Failed to fetch and store data', details: error.message });
  }
});

// Basic health-check
app.get('/health-check', (req, res) => {
  res.status(200).send('Server is up and running.');
});

// ========== 8) Start the server ==========
// ========== 8) Start the server ==========
async function startServer() {
  try {
    await client.connect();
    console.log('✅ Connected to MongoDB');

    // Schedule cron job to run twice daily at 8 AM and 8 PM
    cron.schedule('0 8,20 * * *', async () => {
      console.log('🕒 Running scheduled data fetch:', new Date().toISOString());
      try {
        await fetchAndStoreData();
        console.log('✅ Scheduled fetch completed successfully');
      } catch (error) {
        console.error('❌ Scheduled fetch failed:', error);
      }
    });

    // Initial fetch
    await fetchAndStoreData();

    // Start Express with fallback if port is in use
    await (async function listenWithFallback(port) {
      try {
        await new Promise((resolve, reject) => {
          const server = app.listen(port, () => resolve());
          server.on('error', (err) => reject(err));
        });
        console.log(`🚀 Server running on http://localhost:${port}`);
      } catch (err) {
        if (err && err.code === 'EADDRINUSE') {
          const explicitPort = Boolean(process.env.PORT);
          if (explicitPort) {
            console.error(`❌ Port ${port} is already in use. Set a different PORT or free this port.`);
            throw err;
          } else {
            const fallbackPort = port + 1;
            console.warn(`⚠️ Port ${port} in use. Trying fallback port ${fallbackPort}...`);
            await new Promise((resolve, reject) => {
              const server = app.listen(fallbackPort, () => resolve());
              server.on('error', (e) => reject(e));
            });
            console.log(`🚀 Server running on http://localhost:${fallbackPort}`);
          }
        } else {
          throw err;
        }
      }
    })(PORT);
  } catch (error) {
    console.error('❌ Failed to start server:', error.message);
    process.exit(1);
  }
}
// Graceful shutdown
function shutdown() {
  console.log('🛑 Shutting down server...');
  client.close(false)
    .then(() => {
      console.log('🗄 MongoDB connection closed.');
      process.exit(0);
    })
    .catch((err) => {
      console.error('❌ Error closing MongoDB:', err.message);
      process.exit(1);
    });
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
process.on('unhandledRejection', (reason, promise) => {
  console.error('⚠️ Unhandled Rejection at:', promise, 'reason:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('⚠️ Uncaught Exception thrown:', err.message);
});

startServer();
