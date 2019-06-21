import React from "react";
import PropTypes from "prop-types";
// @material-ui/core components
import {  Table, TableHead, TableCell, TableBody, TableRow, IconButton } from '@material-ui/core';
import withStyles from "@material-ui/core/styles/withStyles";
import { Pageview } from '@material-ui/icons';
// core components
import tableStyle from "assets/jss/material-dashboard-react/components/tableStyle.jsx";
import * as moment from 'moment';
import { AppRoute } from '../../app/AppNavigator';

function CustomTable({ ...props }) {
  const { classes, tableHead, tableData, tableHeaderColor } = props;
  const { appUtils: { appNavigator } } = props

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
                <TableCell padding="none">
                  <IconButton onClick={() => {
                    const link = AppRoute.TRIAL_DETAIL
                                 .replace(':trialId', x.id)
                    appNavigator.goTo(link);
                  }}>
                    <Pageview />
                  </IconButton>
                </TableCell>
                <TableCell>{x.id}</TableCell>
                <TableCell>{x.model_name}</TableCell>
                <TableCell>{x.score}</TableCell>
                <TableCell>{x.status}</TableCell>
                <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                <TableCell>{x.datetime_stopped ? moment(x.datetime_stopped).fromNow() : '-'}</TableCell>
                <TableCell>{
                  x.datetime_stopped ?
                    // @ts-ignore
                    moment.duration(x.datetime_stopped - x.datetime_started).humanize()
                    : '-'
                }</TableCell>
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
