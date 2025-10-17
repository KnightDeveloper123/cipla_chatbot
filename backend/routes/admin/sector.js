const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');
const multer = require('multer')
const path = require('path');
const { data } = require('react-router-dom');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, '../../sectors'));
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 },
});


router.post('/add', middleware, upload.single('icon'), async (req, res) => {
  try {
    let { name, category, description, admin_id, employee_id } = req.body;
    // if (typeof products === 'string') {
    //   try {
    //     products = JSON.parse(products);
    //   } catch (err) {
    //     return res.status(400).json({ error: "Invalid products format" });
    //   }
    // }
    // if (!Array.isArray(products) || products.length === 0) {
    //   return res.status(400).json({ error: "products must be a non-empty array" });
    // }

    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required.' });
    }

    const file_name = req.file.filename;

    const sql = `INSERT INTO sector (name, category, description, icon,admin_id,employee_id) VALUES (?, ?, ?, ?,?, ?)`;
    const values = [name, category, description, file_name, admin_id, employee_id];

    connection.query(sql, values, (err, result) => {
      if (err) {
        console.error(err);
        return res.status(400).json({ error: "Something went wrong" });
      }

      const sector_id = result.insertId;

      // const productSector = `INSERT INTO product_sector (sector_id, product_id) VALUES ?`;

      // const productValues = products.map(p => [sector_id, p.product_id || p]);

      // connection.query(productSector, [productValues], (assignErr) => {
      //   if (assignErr) {
      //     console.error(assignErr);
      //     return res.status(400).json({ error: "Error assigning products to sector" });
      //   }

      return res.send({ success: "Sector created" });
    });
    // });

  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

router.post('/update', middleware, upload.single('icon'), async (req, res) => {
  try {

    //   console.log(req.body);

    let { name, category, description, products, sector_id, employee_id } = req.body;

    // if (typeof products === 'string') {
    //   try {
    //     products = JSON.parse(products);
    //   } catch (err) {
    //     return res.status(400).json({ error: "Invalid products format" });
    //   }
    // }

    // if (!Array.isArray(products) || products.length === 0) {
    //   return res.status(400).json({ error: "Products must be a non-empty array" });
    // }

    let updateFields = [];
    let updateValues = [];

    if (name) {
      updateFields.push("name = ?");
      updateValues.push(name);
    }
    if (category) {
      updateFields.push("category = ?");
      updateValues.push(category);
    }
    if (description) {
      updateFields.push("description = ?");
      updateValues.push(description);
    }
    if (employee_id) {
      updateFields.push("employee_id = ?");
      updateValues.push(employee_id);
    }
    if (req.file) {
      updateFields.push("icon = ?");
      updateValues.push(req.file.filename);
    }

    if (updateFields.length === 0) {
      return res.status(400).json({ error: "No fields to update" });
    }

    const updateSql = `UPDATE sector SET ${updateFields.join(', ')} WHERE id = ?`;
    updateValues.push(sector_id);

    connection.query(updateSql, updateValues, (updateErr) => {
      if (updateErr) {
        console.error(updateErr);
        return res.status(400).json({ error: "Error updating sector" });
      }

      // Delete old product_sector relations
      // const deleteSql = `DELETE FROM product_sector WHERE sector_id = ?`;
      // connection.query(deleteSql, [sector_id], (deleteErr) => {
      //   if (deleteErr) {
      //     console.error(deleteErr);
      //     return res.status(400).json({ error: "Error clearing old product-sector mappings" });
      //   }

      // Insert new product_sector relations
      // const insertSql = `INSERT INTO product_sector (sector_id, product_id) VALUES ?`;
      // const productValues = products.map(p => [sector_id, p.product_id || p]);

      // connection.query(insertSql, [productValues], (insertErr) => {
      //   if (insertErr) {
      //     console.error(insertErr);
      //     return res.status(400).json({ error: "Error updating products in sector" });
      //   }

      // });
      return res.send({ success: "Sector updated successfully" });
      // });
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

router.post('/delete', middleware, (req, res) => {
  try {
    const { sector_id } = req.body
    connection.query(`update sector set status='1', updated_on=NOW() where id=?`, [sector_id], (err, data) => {
      if (err) {
        console.log(err);
        res.status(400).json({ error: "Something went wrong!" })
      } else {
        res.json({ data, success: "Sector Deleted!" })
      }
    })
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Internal server error!" })
  }
})




router.post('/addCategory', async (req, res) => {
  try {
    let { name } = req.body;
    const sql = `INSERT INTO category (name) VALUES (?)`;
    const values = [name];

    connection.query(sql, values, (err, result) => {
      if (err) {
        console.error(err);
        return res.status(400).json({ error: "Something went wrong" });
      }
      return res.send({ success: "Category added", result });
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/get_all_category', async (req, res) => {

  try {

    let result = await executeQuery(`select * from category where status=0`)
    res.json({ result })
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Internal server error!" })
  }
})

// GET BY PRODUCT

router.get('/sectort_by_id', middleware, async (req, res) => {
  try {
    const { sector_id } = req.query
    const data = await executeQuery(`select  * from sector where id=${sector_id}`)
    return res.json({ data: data[0], })
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Internal server error!" })
  }
})

// GET ALL PRODUCT
router.get('/get_all_sector', middleware, async (req, res) => {
  try {
    const { admin_id } = req.query
    const data = await executeQuery(` SELECT
        sector.id,
        sector.name,
        sector.status,
        sector.created_on,
        sector.category,
        sector.description,
        sector.icon,
        sector.status,
        employee.name as emp_name,
        employee.id as employee_id,
        COUNT(bots.id) AS bot_count,
        GROUP_CONCAT(DISTINCT product_service.sector_id) AS psId
      FROM sector
      LEFT JOIN bots ON bots.sector_id = sector.id
      LEFT JOIN product_service ON product_service.sector_id = sector.id
      LEFT JOIN employee ON sector.employee_id = employee.id
      WHERE sector.status = 0 AND sector.admin_id = ${admin_id}
      
      GROUP BY
        sector.id,
        sector.name,
        sector.status,
        sector.created_on,
        sector.category,
        sector.description,
        sector.icon,
        sector.status,
        employee.name,
        employee.id
      ORDER BY sector.id DESC;`)

    const formattedData = data.map(sector => ({
      ...sector,
      product_ids: sector.psId ? sector.psId.split(',').map(Number) : []
    }));
    // console.log('data', formattedData)
    return res.json({ formattedData })
  } catch (error) {
    console.log(error);
    return res.status(500).json({ error: "Internal server error!" })
  }
})


router.get('/get_all_product_sector', middleware, async (req, res) => {
  try {
    const { sector_id } = req.query
    const data = await executeQuery(`SELECT 
    sector.*, 
    ps.name AS product_name,
    ps.description AS product_description,
    ps.id AS product_id,
    ps.image
  FROM sector
  LEFT JOIN product_service AS ps ON ps.sector_id = sector.id
           where sector.id=${sector_id}`)
    return res.json({ data })

  } catch (error) {
    return res.status(500).json({ error: "Internal server error!" })
  }
})


router.get('/get_linked_bot', middleware, async (req, res) => {
  const { sector_id } = req.query;

  if (!sector_id) {
    return res.status(400).json({ error: "sector_id is required" });
  }

  try {
    const result = await executeQuery(
      `
      SELECT 
        s.*, 
        (
          SELECT JSON_ARRAYAGG(
            JSON_OBJECT(
              'id', b.id, 
              'name', b.name
            )
          )
          FROM bots b
          WHERE b.sector_id = s.id AND b.status = 0
        ) AS bots
      FROM sector s
      WHERE s.id = ${sector_id}
      GROUP BY s.id
      `
    );

   if (result.length === 0) {
      console.log("⚠️ No sector found with the given sector_id.");
      return res.json({ data: { bots: [] } });
    }

    const botsField = result[0].bots;

    if (typeof botsField === 'string') {
      console.log("🔍 bots field is a string. Attempting to parse.");
    } else {
      console.log("✅ bots field is already parsed.");
    }

    const parsedBots = typeof botsField === 'string' ? JSON.parse(botsField) : botsField;

    // console.log("✅ Parsed bots array:", parsedBots);
  return res.json({
      data: {
        ...result[0],
        bots: parsedBots
      }
    });
  } catch (error) {
    console.error("❌ Error fetching linked bots:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});


module.exports = router;