import React from "react";
import PropTypes from "prop-types";
// @material-ui/core components
import {  Table, TableHead, TableCell, TableBody, TableRow } from '@material-ui/core';
import withStyles from "@material-ui/core/styles/withStyles";

// core components
import tableStyle from "assets/jss/material-dashboard-react/components/tableStyle.jsx";
import * as moment from 'moment';

function CustomTable({ ...props }) {
  const { classes, tableHead, tableData, tableHeaderColor } = props;

  return (
    <div className={classes.tableResponsive}>
      <Table className={classes.table}>
        {tableHead !== undefined ? (
          <TableHead className={classes[tableHeaderColor + "TableHeader"]}>
            <TableRow>
              {tableHead.map((prop, key) => {
                return (
                  <TableCell
                    className={classes.tableCell + " " + classes.tableHeadCell}
                    key={key}
                  >
                    {prop}
                  </TableCell>
                );
              })}
            </TableRow>
          </TableHead>
        ) : null}
        <TableBody>
          {tableData.map(x => {
            return (
              <TableRow key={x.id} hover>
                <TableCell>{x.id}</TableCell>
                <TableCell>{x.name}</TableCell>
                <TableCell>{x.task}</TableCell>
                <TableCell>{x.size_bytes} bytes</TableCell>
                <TableCell>{moment(x.datetime_created).fromNow()}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

CustomTable.defaultProps = {
  tableHeaderColor: "gray"
};

CustomTable.propTypes = {
  classes: PropTypes.object.isRequired,
  tableHeaderColor: PropTypes.oneOf([
    "warning",
    "primary",
    "danger",
    "success",
    "info",
    "rose",
    "gray"
  ]),
  tableHead: PropTypes.arrayOf(PropTypes.string),
  tableData: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.string))
};

export default withStyles(tableStyle)(CustomTable);
