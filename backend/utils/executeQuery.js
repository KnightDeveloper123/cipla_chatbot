const connection = require('../database/db');


function executeQuery(query, values) {
    return new Promise((resolve, reject) => {
        connection.query(query, values, (err, results) => {
            if (err) {
                console.log(err);
                reject(err);
            } else {
                // console.log(results)
                resolve(results);
            }
        });
    });
}


module.exports = executeQuery